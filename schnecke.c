/*
    Schnecke LLM Inference Engine for AmigaOS
    Based on llama2.c by Andrej Karpathy

    Original llama2.c:
    Copyright (c) 2023 Andrej Karpathy

    AmigaOS/68k port, endian handling,
    C89 compatibility changes and other modifications:
    Copyright (c) 2026 Jani Kuituniemi

    Licensed under the MIT License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    int dim; /*  transformer dimension */
    int hidden_dim; /*  for ffn layers */
    int n_layers; /*  number of layers */
    int n_heads; /*  number of query heads */
    int n_kv_heads; /*  number of key/value heads (can be < query heads because of multiquery) */
    int vocab_size; /*  vocabulary size, usually 256 (byte-level) */
    int seq_len; /*  max sequence length */
} Config;

typedef struct {
    /*  token embedding table */
    float* token_embedding_table;    /*  (vocab_size, dim) */
    /*  weights for rmsnorms */
    float* rms_att_weight; /*  (layer, dim) rmsnorm weights */
    float* rms_ffn_weight; /*  (layer, dim) */
    /*  weights for matmuls. note dim == n_heads * head_size */
    float* wq; /*  (layer, dim, n_heads * head_size) */
    float* wk; /*  (layer, dim, n_kv_heads * head_size) */
    float* wv; /*  (layer, dim, n_kv_heads * head_size) */
    float* wo; /*  (layer, n_heads * head_size, dim) */
    /*  weights for ffn */
    float* w1; /*  (layer, hidden_dim, dim) */
    float* w2; /*  (layer, dim, hidden_dim) */
    float* w3; /*  (layer, hidden_dim, dim) */
    /*  final rmsnorm */
    float* rms_final_weight; /*  (dim,) */
    /*  (optional) classifier weights for the logits, on the last layer */
    float* wcls;
} TransformerWeights;

typedef struct {
    /*  current wave of activations */
    float *x; /*  activation at current time stamp (dim,) */
    float *xb; /*  same, but inside a residual branch (dim,) */
    float *xb2; /*  an additional buffer just for convenience (dim,) */
    float *hb; /*  buffer for hidden dimension in the ffn (hidden_dim,) */
    float *hb2; /*  buffer for hidden dimension in the ffn (hidden_dim,) */
    float *q; /*  query (dim,) */
    float *k; /*  key (dim,) */
    float *v; /*  value (dim,) */
    float *att; /*  buffer for scores/attention values (n_heads, seq_len) */
    float *logits; /*  output logits */
    /*  kv cache */
    float* key_cache;   /*  (layer, seq_len, dim) */
    float* value_cache; /*  (layer, seq_len, dim) */
} RunState;

typedef struct {
    Config config; /*  the hyperparameters of the architecture (the blueprint) */
    TransformerWeights weights; /*  the weights of the model */
    RunState state; /*  buffers for the "wave" of activations in the forward pass */
    int fd; /*  file descriptor for memory mapping */
    float* data; /*  memory mapped data pointer */
    ssize_t file_size; /*  size of the checkpoint file in bytes */
} Transformer;

int host_is_little_endian(void) {
    unsigned int x = 1;
    return *((unsigned char*)&x);
}

unsigned int bswap_u32(unsigned int x) {
    return ((x & 0x000000FFU) << 24) |
           ((x & 0x0000FF00U) << 8)  |
           ((x & 0x00FF0000U) >> 8)  |
           ((x & 0xFF000000U) >> 24);
}

unsigned int le32_to_host_u32(unsigned int x) {
    if (host_is_little_endian()) {
        return x;
    }
    return bswap_u32(x);
}

int le32_to_host_i32(int x) {
    unsigned int ux = (unsigned int)x;
    ux = le32_to_host_u32(ux);
    return (int)ux;
}

float le32_bytes_to_host_f32(unsigned char bytes[4]) {
    union {
        unsigned int u;
        float f;
    } v;
    v.u = (unsigned int)bytes[0] |
          ((unsigned int)bytes[1] << 8) |
          ((unsigned int)bytes[2] << 16) |
          ((unsigned int)bytes[3] << 24);
    return v.f;
}

int read_le_i32(FILE *file, int *out) {
    int x;
    if (fread(&x, sizeof(int), 1, file) != 1) {
        return 0;
    }
    *out = le32_to_host_i32(x);
    return 1;
}

int read_le_u32(FILE *file, unsigned int *out) {
    unsigned int x;
    if (fread(&x, sizeof(unsigned int), 1, file) != 1) {
        return 0;
    }
    *out = le32_to_host_u32(x);
    return 1;
}

int read_le_f32(FILE *file, float *out) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, file) != 4) {
        return 0;
    }
    *out = le32_bytes_to_host_f32(bytes);
    return 1;
}

void malloc_run_state(RunState* s, Config* p) {
    /*  we calloc instead of malloc to keep valgrind happy */
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    /*  ensure all mallocs went fine */
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    /*  make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models */
    unsigned long n_layers = (unsigned long)p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; /*  skip what used to be freq_cis_real (for RoPE) */
    ptr += p->seq_len * head_size / 2; /*  skip what used to be freq_cis_imag (for RoPE) */
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    int shared_weights;
    float* weights_ptr;
    long weight_bytes;
    long n_weights;
    long i;
    unsigned char raw[4];
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    /*  read in the config header */
    if (!read_le_i32(file, &config->dim) ||
        !read_le_i32(file, &config->hidden_dim) ||
        !read_le_i32(file, &config->n_layers) ||
        !read_le_i32(file, &config->n_heads) ||
        !read_le_i32(file, &config->n_kv_heads) ||
        !read_le_i32(file, &config->vocab_size) ||
        !read_le_i32(file, &config->seq_len)) { exit(EXIT_FAILURE); }
    /*  negative vocab size is hacky way of signaling unshared weights. */
    shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    /*  figure out the file size */
    fseek(file, 0, SEEK_END); /*  move file pointer to end of file */
    *file_size = ftell(file); /*  get the file size, in bytes */
    fclose(file);

    file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    if (fseek(file, sizeof(Config), SEEK_SET) != 0) {
        fprintf(stderr, "fseek failed!\n");
        exit(EXIT_FAILURE);
    }

    weight_bytes = (long)(*file_size - (ssize_t)sizeof(Config));

    if (weight_bytes < 0 || (weight_bytes % 4) != 0) {
        fprintf(stderr, "bad checkpoint size\n");
        exit(EXIT_FAILURE);
    }

    n_weights = weight_bytes / 4;
    weights_ptr = (float*)malloc((size_t)n_weights * sizeof(float));

    if (weights_ptr == NULL) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < n_weights; i++) {
        if (fread(raw, 1, 4, file) != 4) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        weights_ptr[i] = le32_bytes_to_host_f32(raw);
    }

    fclose(file);

    *data = weights_ptr;
    *fd = -1;

    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    /* read in the Config and the Weights from the checkpoint */
    read_checkpoint(checkpoint_path,
                    &t->config,
                    &t->weights,
                    &t->fd,
                    &t->data,
                    &t->file_size);

    /* allocate the RunState buffers */
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->data != NULL) {
        free(t->data);
    }
    /*  free the RunState buffers */
    free_run_state(&t->state);
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    /*  calculate sum of squares */
    float ss = 0.0f;
    int j;
    for (j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    /*  normalize and scale */
    for (j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    /*  find max value (for numerical stability) */
    float max_val = x[0];
    int i;
    float sum = 0.0f;
    for (i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    /*  exp and sum */
    for (i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    /*  normalize */
    for (i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    /* W (d,n) @ x (n,) -> xout (d,) */
    int i, j;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p;
    TransformerWeights* w;
    RunState* s;
    float *x;
    int dim, kv_dim, kv_mul, hidden_dim, head_size;
    int l, i, h, t, v;
    int head_dim, rotn, loff;
    float freq, val, fcr, fci;
    float *content_row;
    float *key_cache_row;
    float *value_cache_row;
    float *q;
    float *att;
    float *k;
    float *vptr;
    float *xb;
    float score;
    float a;
    float v0, v1;
    float *vec;

    p = &transformer->config;
    w = &transformer->weights;
    s = &transformer->state;
    x = s->x;

    dim = p->dim;
    kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    kv_mul = p->n_heads / p->n_kv_heads;
    hidden_dim = p->hidden_dim;
    head_size = dim / p->n_heads;

    content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    for (l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (i = 0; i < dim; i += 2) {
            head_dim = i % head_size;
            freq = 1.0f / pow(10000.0, head_dim / (float)head_size);
            val = pos * freq;
            fcr = cos(val);
            fci = sin(val);
            rotn = i < kv_dim ? 2 : 1;

            for (v = 0; v < rotn; v++) {
                vec = v == 0 ? s->q : s->k;
                v0 = vec[i];
                v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        loff = l * p->seq_len * kv_dim;
        key_cache_row = s->key_cache + loff + pos * kv_dim;
        value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        for (h = 0; h < p->n_heads; h++) {
            q = s->q + h * head_size;
            att = s->att + h * p->seq_len;

            for (t = 0; t <= pos; t++) {
                k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                score = 0.0f;
                for (i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrt((float)head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));

            for (t = 0; t <= pos; t++) {
                vptr = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                a = att[t];
                for (i = 0; i < head_size; i++) {
                    xb[i] += a * vptr[i];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        for (i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        for (i = 0; i < hidden_dim; i++) {
            val = s->hb[i];
            val *= (1.0f / (1.0f + exp(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        for (i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; /*  stores all single-byte strings */
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    int i;
    int len;
    FILE *file;
    t->vocab_size = vocab_size;
    /*  malloc space to hold the scores and the strings */
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; /*  initialized lazily */
    for (i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    /*  read in the file */
    file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (!read_le_u32(file, &t->max_token_length)) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    for (i = 0; i < vocab_size; i++) {
        if (!read_le_f32(file, t->vocab_scores + i)) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (!read_le_i32(file, &len)) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; /*  add the string terminating token */
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    int i;
    for (i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    unsigned char byte_val;
    /*  following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89) */
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    /*  Some tokens designate raw bytes, and look like e.g. '<0x01>' */
    /*  parse this and convert and return the actual byte */
    if (sscanf(piece, "<0x%2X>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    /*  piece might be a raw byte token, and we only want to print printable chars or whitespace */
    /*  because some of the other bytes can be various control codes, backspace, etc. */
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; /*  bad byte, do not print it */
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    /*  Find the perfect match for str in vocab, return its index or -1 if not found */
    TokenIndex tok;
    TokenIndex *res;
    tok.str = str; /*  acts as the key to search for */
    tok.id = 0;
    res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    /*  encode the string text (input) into an upper-bound pre-allocated tokens[] array */
    /*  bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2) */
    int i;
    char* str_buffer;
    size_t str_len = 0;
    char *c;
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        /*  lazily malloc and sort the vocabulary */
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    /*  create a temporary buffer that will store merge candidates of always two consecutive tokens */
    /*  *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1) */
    str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));

    /*  start at 0 tokens */
    *n_tokens = 0;

    /*  add optional BOS (=1) token, if desired */
    if (bos) tokens[(*n_tokens)++] = 1;

    /*  add_dummy_prefix is true by default */
    /*  so prepend a dummy prefix token to the input string, but only if text != "" */
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    /*  process the raw (UTF-8) byte sequence of the input string */
    {
    for (c = text; *c != '\0'; c++) {
        int id;

        /*  reset buffer if the current byte is ASCII or a leading byte */
        /*  0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest */
        /*  0x80 is 10000000 */
        /*  in UTF-8, all continuation bytes start with "10" in first two bits */
        if ((*c & 0xC0) != 0x80) {
            /*  this byte must be either a leading byte (11...) or an ASCII char (0x...) */
            /*  => reset our location, as we are starting a new UTF-8 codepoint */
            str_len = 0;
        }

        /*  append the current byte to the buffer */
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        /*  while the next character is a continuation byte, continue appending */
        /*  but if there are too many of them, just stop to avoid overruning str_buffer size. */
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        /*  ok c+1 is not a continuation byte, so we have read in a full codepoint */
        id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            /*  we found this codepoint in vocab, add it as a token */
            tokens[(*n_tokens)++] = id;
        } else {
            /*  byte_fallback encoding: just encode each byte as a token */
            /*  +3 is here because the first 3 vocab elements are <unk>, <s>, </s> */
            /*  so the individual bytes only start at index 3 */
            int i;
            for (i=0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; /*  protect against a sequence of stray UTF8 continuation bytes */
      }
    }

    /*  merge the best consecutive pair each iteration, according the scores in vocab_scores */
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        int i;
        int id;

        for (i=0; i < (*n_tokens-1); i++) {
            /*  check if we can merge the pair (tokens[i], tokens[i+1]) */
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                /*  this merge pair exists in vocab! record its score and position */
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; /*  we could not find any more pairs to merge, so we are done */
        }

        /*  merge the consecutive pair (best_idx, best_idx+1) into new token best_id */
        tokens[best_idx] = best_id;
        /*  delete token at position best_idx+1, shift the entire sequence back 1 */
        for (i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; /*  token length decreased */
    }

    /*  add optional EOS (=2) token, if desired */
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

typedef struct {
    float prob;
    int index;
} ProbIndex; /*  struct used when sorting probabilities during top-p sampling */

typedef struct {
    int vocab_size;
    ProbIndex* probindex; /*  buffer used in top-p sampling */
    float temperature;
    float topp;
    unsigned long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    /*  return the index that has the highest probability */
    int max_i = 0;
    float max_p = probabilities[0];
    int i;
    for (i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    /*  sample index from probabilities (they must sum to 1!) */
    /*  coin is a random number in [0, 1), usually from random_f32() */
    float cdf = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; /*  in case of rounding errors */
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    /*  top-p sampling (or "nucleus sampling") samples from the smallest set of */
    /*  tokens that exceed probability topp. This way we never sample tokens that */
    /*  have very low probabilities and are less likely to go "off the rails". */
    /*  coin is a random number in [0, 1), usually from random_f32() */

    int n0 = 0;
    /*  quicksort indices in descending order of probabilities */
    /*  values smaller than (1 - topp) / (n - 1) cannot be part of the result */
    /*  so for efficiency we crop these out as candidates before sorting */
    const float cutoff = (1.0f - topp) / (n - 1);
    float cumulative_prob = 0.0f;
    int last_idx;
    float r;
    float cdf = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    /*  truncate the list where cumulative probability exceeds topp */
    last_idx = n0 - 1; /*  in case of rounding errors consider all elements */
    for (i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; /*  we have exceeded topp by including last_idx */
        }
    }

    /*  sample from the truncated list */
    r = coin * cumulative_prob;
    for (i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; /*  in case of rounding errors */
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    /*  buffer only used with nucleus sampling; may not need but it is small */
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long *state) { /*  random float32 in [0,1) */
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    /*  sample the token given the logits and some hyperparameters */
    int next;
    if (sampler->temperature == 0.0f) {
        /*  greedy argmax sampling: take the token with the highest probability */
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        float coin;
        /*  apply the temperature to the logits */
        int q;
        for (q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        /*  apply softmax to the logits to get the probabilities for next token */
        softmax(logits, sampler->vocab_size);
        /*  flip a (float) coin (this is our source of entropy for sampling) */
        coin = random_f32(&sampler->rng_state);
        /*  we sample from this distribution to get the next token */
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            /*  simply sample from the predicted probability distribution */
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            /*  top-p (nucleus) sampling, clamping the least likely tokens to zero */
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

long time_in_ms(void) {
    return (long)((clock() * 1000) / CLOCKS_PER_SEC);
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    int num_prompt_tokens = 0;
    int* prompt_tokens;
    long start = 0;  /*  used to time our code, only initialized after first iteration */
    int next;
    int token;
    int pos = 0;
    if (prompt == NULL) { prompt = empty_prompt; }

    /*  encode the (string) prompt into tokens sequence */
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); /*  +3 for '\0', ?BOS, ?EOS */
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    /*  start the main loop */
    token = prompt_tokens[0]; /*  kick off with the first token in the prompt */
    while (pos < steps) {
        char* piece;
        float* logits;

        /*  forward the transformer to get logits for the next token */
        logits = forward(transformer, token, pos);

        /*  advance the state machine */
        if (pos < num_prompt_tokens - 1) {
            /*  if we are still processing the input prompt, force the next prompt token */
            next = prompt_tokens[pos + 1];
        } else {
            /*  otherwise sample the next token from the logits */
            next = sample(sampler, logits);
        }
        pos++;

        /*  data-dependent terminating condition: the BOS (=1) token delimits sequences */
        if (next == 1) { break; }

        /*  print the token as string, decode it with the Tokenizer object */
        piece = decode(tokenizer, token, next);
        safe_printf(piece); /*  same as printf("%s", piece), but skips "unsafe" bytes */
        fflush(stdout);
        token = next;

        /*  init the timer here because the first iteration can be slower */
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    /*  report achieved tok/s (pos-1 because the timer starts after first iteration) */
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Achieved tokens/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   schnecke <checkpoint> [options]\n");
    fprintf(stderr, "Example: schnecke model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -h          This help\n");
    exit(EXIT_FAILURE);
}

void print_banner() {
  printf("Schnecke LLM Inference Engine v1.0.0\n");
  printf("For AmigaOS and classic systems\n\n");
  printf("Originally based on llama2.c by Andrej Karpathy\n");
  printf("AmigaOS/68k port and modifications by Jani Kuituniemi\n\n");
  printf("Build date: May 2026\n\n");
  printf("Run with -h for help.\n\n");
}

int main(int argc, char *argv[]) {
    /*  default parameters */
    char *checkpoint_path = NULL;  /*  e.g. out/model.bin */
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   /*  0.0 = greedy deterministic. 1.0 = original. do not set higher */
    float topp = 0.9f;          /*  top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower */
    int steps = 256;            /*  number of steps to run for */
    char *prompt = "";        /*  prompt string */
    unsigned long rng_seed = 0; /*  seed rng with time by default */
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    /*  simple C argparse so we can override the defaults above from the command line */
    int i;
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0) {
            error_usage();
        }
    }
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (i = 2; i < argc; i+=2) {
        /*  do some basic validation */
        if (i + 1 >= argc) { error_usage(); } /*  must have arg after flag */
        if (argv[i][0] != '-') { error_usage(); } /*  must start with dash */
        if (strlen(argv[i]) != 2) { error_usage(); } /*  must be -x (one dash, one letter) */
        /*  read in the args */
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else { error_usage(); }
    }

    /* Print program banner */
    print_banner();

    /*  parameter validation/overrides */
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    /*  build the Transformer via the model .bin file */
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; /*  override to ~max length */

    /*  build the Tokenizer via the tokenizer .bin file */
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    /*  build the Sampler */
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    /*  run! */
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    /*  memory and file handles cleanup */
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
