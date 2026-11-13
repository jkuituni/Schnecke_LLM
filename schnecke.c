/*
    Schnecke LLM Inference Engine for AmigaOS
    Based on llama2.c by Andrej Karpathy

    Original llama2.c:
    Copyright (c) 2023 Andrej Karpathy

    AmigaOS/68k port, endian handling, C89
    compatibility changes, Q8_0 support and
	other modifications:
    Copyright (c) 2026 Jani Kuituniemi

    Version 1.0.0 - May 2026
    Version 1.1.0 - Nov 2026

    Licensed under the MIT License.
*/

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Model formats and checkpoint file constants */
#define MODEL_FP32 0
#define MODEL_Q8_0 1
#define CHECKPOINT_MAGIC 0x616b3432UL
#define CHECKPOINT_VERSION_Q8 2
#define FP32_HEADER_SIZE 28

/* Model architecture and quantization configuration */
typedef struct Config
{
    int dim;        /* Transformer embedding dimension */
    int hidden_dim; /* Feed-forward network hidden dimension */
    int n_layers;   /* Number of transformer layers */
    int n_heads;    /* Number of attention query heads */
    int n_kv_heads; /* Number of key/value heads (grouped-query attention) */
    int vocab_size; /* Number of tokens in the vocabulary */
    int seq_len;    /* Maximum sequence length */
    int group_size; /* Number of weights per quantization group */
} Config;

/* Quantized tensors used in Q8_0 codepath
   Q8_0 tensor: signed 8-bit values and one FP32 scale per group */
typedef struct QuantizedTensor
{
	signed char *q; /* Quantized int8 values */
	float *s;	/* FP32 scales */
} QuantizedTensor;

/* Model parameters for FP32 and Q8_0 inference.
   Q8_0 quantizes the major weight matrices while retaining
   normalization weights in FP32. */
typedef struct TransformerWeights
{
	/* FP32 weights: */
	/* Token embedding table */
	float *token_embedding_table; /* (vocab_size, dim) */
	/* Weights for rmsnorms */
	float *rms_att_weight; /* (layer, dim) rmsnorm weights */
	float *rms_ffn_weight; /* (layer, dim) */
	/* Weights for matmuls. note dim == n_heads * head_size */
	float *wq; /* (layer, dim, n_heads * head_size) */
	float *wk; /* (layer, dim, n_kv_heads * head_size) */
	float *wv; /* (layer, dim, n_kv_heads * head_size) */
	float *wo; /* (layer, n_heads * head_size, dim) */
	/* Weights for ffn */
	float *w1; /* (layer, hidden_dim, dim) */
	float *w2; /* (layer, dim, hidden_dim) */
	float *w3; /* (layer, hidden_dim, dim) */
	/* Final rmsnorm */
	float *rms_final_weight; /* (dim,) */
	/* (optional) classifier weights for the logits, on the last layer */
	float *wcls;

	/* Q8_0 weights: */
	QuantizedTensor q_token_embedding_table;
	QuantizedTensor *q_wq;
	QuantizedTensor *q_wk;
	QuantizedTensor *q_wv;
	QuantizedTensor *q_wo;
	QuantizedTensor *q_w1;
	QuantizedTensor *q_w2;
	QuantizedTensor *q_w3;
	QuantizedTensor q_wcls;
} TransformerWeights;

/* Runtime buffers used during transformer inference.
   These hold intermediate activations, attention scores,
   cached keys/values and the final output logits. */
typedef struct RunState
{
	/*  Current wave of activations */
	float *x;   /* Activation at current time stamp (dim,) */
	float *xb;  /* Same, but inside a residual branch (dim,) */
	float *xb2; /* An additional buffer just for convenience (dim,) */
	float *hb;  /* Buffer for hidden dimension in the ffn (hidden_dim,) */
	float *hb2; /* Buffer for hidden dimension in the ffn (hidden_dim,) */
	float *q;   /* Query (dim,) */
	float *k;   /* Key (dim,) */
	float *v;   /* Value (dim,) */
	float *att; /* Buffer for scores/attention values (n_heads, seq_len) */
	float *logits; /* Output logits */
	/*  KV cache */
	float *key_cache;   /* (layer, seq_len, dim) */
	float *value_cache; /* (layer, seq_len, dim) */
	/* Quantized activation buffers (Q8_0 only) */
	QuantizedTensor q_x;  /* Quantized dim-sized activation */
	QuantizedTensor q_hb; /* Quantized hidden_dim-sized activation */
} RunState;

/* Complete transformer instance: architecture, model weights,
   runtime state and checkpoint storage. */
typedef struct Transformer
{
	Config config; /* The hyperparameters of the architecture */
	TransformerWeights weights; /* The weights of the model */
	RunState state; /* Buffers for the "wave" of activations in the forward pass */
	int fd;		/* File descriptor for memory mapping */
	void *data;	/* Memory mapped data pointer */
	ssize_t file_size; /* Size of the checkpoint file in bytes */
	int model_format;  /* Model format, fp32 or q8_0 */
} Transformer;

/* Vocabulary entry used for efficient token lookup */
typedef struct TokenIndex
{
	char *str;
	int id;
} TokenIndex;

/* Tokenizer vocabulary, token scores and lookup data */
typedef struct Tokenizer
{
	char **vocab;
	float *vocab_scores;
	TokenIndex *sorted_vocab;
	int vocab_size;
	unsigned int max_token_length;
	unsigned char byte_pieces[512]; /* Stores all single-byte strings */
} Tokenizer;

/* Token probability and vocabulary index used in top-p sampling */
typedef struct ProbIndex
{
	float prob;
	int index;
} ProbIndex; 

/* Sampling parameters and random number generator state */
typedef struct Sampler
{
	int vocab_size;
	ProbIndex *probindex;
	float temperature;
	float topp;
	unsigned long rng_state;
} Sampler;

/* Check for whether or not we're running in Little or Big Endian host */
int host_is_little_endian(void)
{
	unsigned int x = 1;
	return *((unsigned char *)&x);
}

/* ----- ENDIANNESS HANDLING ROUTINES ----- */

/* Swap the byte order of a 32-bit unsigned integer. */
unsigned int bswap_u32(unsigned int x)
{
	return ((x & 0x000000FFU) << 24) | ((x & 0x0000FF00U) << 8) |
	       ((x & 0x00FF0000U) >> 8) | ((x & 0xFF000000U) >> 24);
}

/* Convert a little-endian unsigned 32-bit value to host byte order. */
unsigned int le32_to_host_u32(unsigned int x)
{
	if (host_is_little_endian())
	{
		return x;
	}
	return bswap_u32(x);
}

/* Convert a little-endian signed 32-bit value to host byte order. */
int le32_to_host_i32(int x)
{
	unsigned int ux = (unsigned int)x;
	ux = le32_to_host_u32(ux);
	return (int)ux;
}

/* Decode a little-endian IEEE-754 FP32 value from four bytes. */
float le32_bytes_to_host_f32(unsigned char bytes[4])
{
	union
	{
		unsigned int u;
		float f;
	} v;
	v.u = (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8) |
	      ((unsigned int)bytes[2] << 16) | ((unsigned int)bytes[3] << 24);
	return v.f;
}

/* Read little-endian 32-bit values from a checkpoint file.
   Return 1 on success, 0 on read failure. */
int read_le_i32(FILE *file, int *out)
{
	int x;
	if (fread(&x, sizeof(int), 1, file) != 1)
	{
		return 0;
	}
	*out = le32_to_host_i32(x);
	return 1;
}

int read_le_u32(FILE *file, unsigned int *out)
{
	unsigned int x;
	if (fread(&x, sizeof(unsigned int), 1, file) != 1)
	{
		return 0;
	}
	*out = le32_to_host_u32(x);
	return 1;
}

int read_le_f32(FILE *file, float *out)
{
	unsigned char bytes[4];
	if (fread(bytes, 1, 4, file) != 4)
	{
		return 0;
	}
	*out = le32_bytes_to_host_f32(bytes);
	return 1;
}
/* ----- END OF ENDIANNESS HANDLING ROUTINES ----- */

/* Detect checkpoint format from the first 32-bit value.
   Q8_0 checkpoints have a magic number; legacy FP32 checkpoints
   have no magic header and are assumed when it is absent.
   Returns -1 on read or seek failure. */
static int detect_model_format(FILE *file)
{
	unsigned int magic;
	if (fseek(file, 0L, SEEK_SET) != 0)
	{
		return -1;
	}
	if (!read_le_u32(file, &magic))
	{
		return -1;
	}
	if (fseek(file, 0L, SEEK_SET) != 0)
	{
		return -1;
	}
	if (magic == CHECKPOINT_MAGIC)
	{
		return MODEL_Q8_0;
	}
	return MODEL_FP32;
}

/* Calculate the required memory for the FP32 running state 
   and allocate it. */
void malloc_run_state(RunState *s, Config *p)
{
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

	/* Initialize Q8 activation pointers for safe cleanup */
	s->q_x.q = NULL;
	s->q_x.s = NULL;
	s->q_hb.q = NULL;
	s->q_hb.s = NULL;

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
	if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k ||
	    !s->v || !s->key_cache || !s->value_cache || !s->att || !s->logits)
	{
		fprintf(stderr, "malloc failed!\n");
		exit(EXIT_FAILURE);
	}
}

/* Calculate the required memory for the Q8_0 running state 
   and allocate it. */
void malloc_run_state_q8(RunState *s, Config *p)
{
	size_t dim;
	size_t hidden_dim;
	size_t groups_x;
	size_t groups_hb;

	if (s == NULL || p == NULL)
	{
		fprintf(stderr, "Invalid Q8 runtime allocation arguments\n");
		exit(EXIT_FAILURE);
	}

	if (p->dim <= 0 || p->hidden_dim <= 0 || p->group_size <= 0)
	{
		fprintf(stderr, "Invalid Q8 runtime dimensions\n");
		exit(EXIT_FAILURE);
	}

	dim = (size_t)p->dim;
	hidden_dim = (size_t)p->hidden_dim;

	/* Allow a partial final activation group */
	groups_x = dim / (size_t)p->group_size;
	if (dim % (size_t)p->group_size != 0)
	{
		groups_x++;
	}

	groups_hb = hidden_dim / (size_t)p->group_size;
	if (hidden_dim % (size_t)p->group_size != 0)
	{
		groups_hb++;
	}

	/* Check scale allocation sizes */
	if (groups_x > (size_t)-1 / sizeof(float) ||
	    groups_hb > (size_t)-1 / sizeof(float))
	{
		fprintf(stderr, "Q8 activation scale allocation overflow\n");
		exit(EXIT_FAILURE);
	}

	/* Allocate quantized activation buffers */
	s->q_x.q = calloc(dim, sizeof(signed char));
	s->q_x.s = calloc(groups_x, sizeof(float));
	s->q_hb.q = calloc(hidden_dim, sizeof(signed char));
	s->q_hb.s = calloc(groups_hb, sizeof(float));

	if (s->q_x.q == NULL || s->q_x.s == NULL || s->q_hb.q == NULL ||
	    s->q_hb.s == NULL)
	{
		fprintf(stderr, "Q8 activation buffer allocation failed\n");
		exit(EXIT_FAILURE);
	}
}

/* Free allocated memory */
void free_run_state(RunState *s)
{
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
	/* Free quantized activation buffers */
	free(s->q_x.q);
	free(s->q_x.s);
	free(s->q_hb.q);
	free(s->q_hb.s);
}

/* Assign FP32 tensor pointers within the contiguous weight buffer.
   Tensor order follows the legacy llama2.c checkpoint layout.
   The buffer must contain all weights described by the config. */
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr,
			int shared_weights)
{
	int head_size = p->dim / p->n_heads;
	unsigned long n_layers = (unsigned long)p->n_layers;
	w->token_embedding_table = ptr;
	ptr += (unsigned long)p->vocab_size * p->dim;
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
	/* Skip the two precomputed RoPE frequency tables stored in the
       legacy checkpoint. RoPE values are calculated during inference. */
	ptr += p->seq_len * head_size / 2; 
	ptr += p->seq_len * head_size / 2;
	/* Weight tying: Re-use token embeddings for the output classifier
       unless the checkpoint contains a separate classifier matrix. */
	w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

/* Calculate Q8_0 tensor storage requirements.
   Each group contains group_size int8 weights and one FP32 scale.
   The current checkpoint format requires complete groups.
   Returns 0 for invalid sizes or arithmetic overflow. */
static int q8_tensor_size(size_t elements, size_t group_size,
			  size_t *weight_bytes, size_t *scale_bytes)
{
	size_t scale_count;
	if (weight_bytes == NULL || scale_bytes == NULL)
	{
		return 0;
	}
	if (group_size == 0 || elements == 0)
	{
		return 0;
	}
	/* Every quantization group must be complete */
	if (elements % group_size != 0)
	{
		return 0;
	}
	scale_count = elements / group_size;
	/* Check for overflow before calculating scale bytes */
	if (scale_count > (size_t)-1 / sizeof(float))
	{
		return 0;
	}
	*weight_bytes = elements * sizeof(signed char);
	*scale_bytes = scale_count * sizeof(float);
	return 1;
}

/* Map a quantized tensor within the loaded Q8_0 payload.
   Validate its size and bounds, assign pointers to the int8
   weights and FP32 scales, and convert scales to host byte order.
   Scale storage must be suitably aligned for float access. */
static int map_q8_tensor(QuantizedTensor *tensor, unsigned char *data,
			 size_t data_size, size_t *offset, size_t elements,
			 size_t group_size)
{
	size_t weight_bytes;
	size_t scale_bytes;
	size_t scale_offset;
	size_t i;
	unsigned char *raw;

	if (tensor == NULL || data == NULL || offset == NULL)
	{
		return 0;
	}

	/* Calculate and validate tensor sizes */
	if (!q8_tensor_size(elements, group_size, &weight_bytes, &scale_bytes))
	{
		return 0;
	}

	/* Check available payload space before advancing */
	if (*offset > data_size || weight_bytes > data_size - *offset ||
	    scale_bytes > data_size - *offset - weight_bytes)
	{
		return 0;
	}

	scale_offset = *offset + weight_bytes;

	/* Zero-copy mapping requires naturally aligned FP32 scales.
	   Both the allocation and the scale region must be
	   suitably aligned for native floating-point access. */
	if (scale_offset % sizeof(float) != 0)
	{
		return 0;
	}

	/* Map quantized weights */
	tensor->q = (signed char *)(data + *offset);

	/* Map FP32 scales */
	tensor->s = (float *)(data + scale_offset);

	/* Convert little-endian scales to native FP32 */
	raw = data + scale_offset;

	for (i = 0; i < scale_bytes / sizeof(float); i++)
	{
		float value = le32_bytes_to_host_f32(raw + i * sizeof(float));
		memcpy(raw + i * sizeof(float), &value, sizeof(float));
	}

	/* Advance to the next tensor */
	*offset = scale_offset + scale_bytes;

	return 1;
}

/*
   Quantize FP32 activations into grouped Q8_0 values.
 
  Each group contains group_size signed 8-bit values
  and one FP32 scaling factor.
 
  The caller owns the output buffers.
   - out->q must hold n signed bytes.
   - out->s must hold ceil(n / group_size) FP32 scales. */
void quantize_q8(QuantizedTensor *out, const float *in, size_t n,
		 size_t group_size)
{
	size_t group;
	size_t i;
	size_t offset;
	size_t num_groups;
	size_t group_len;

	float max_abs;
	float value;
	float scale;
	float scaled;

	int quantized;

	if (out == NULL || in == NULL || out->q == NULL || out->s == NULL)
	{
		fprintf(stderr, "Invalid Q8 quantization buffers\n");
		exit(EXIT_FAILURE);
	}

	if (group_size == 0 || n == 0)
	{
		fprintf(stderr, "Invalid Q8 quantization dimensions\n");
		exit(EXIT_FAILURE);
	}

	/* Round up to include a partial final group */
	num_groups = n / group_size;

	if (n % group_size != 0)
	{
		num_groups++;
	}

	for (group = 0; group < num_groups; group++)
	{
		offset = group * group_size;

		/* Determine the actual number of elements in this group */
		group_len = n - offset;

		if (group_len > group_size)
		{
			group_len = group_size;
		}

		max_abs = 0.0f;

		/* Find the largest absolute value in this group */
		for (i = 0; i < group_len; i++)
		{
			value = in[offset + i];

			if (value < 0.0f)
			{
				value = -value;
			}

			if (value > max_abs)
			{
				max_abs = value;
			}
		}

		/* Handle groups containing only zeros */
		if (max_abs == 0.0f)
		{
			out->s[group] = 0.0f;

			for (i = 0; i < group_len; i++)
			{
				out->q[offset + i] = 0;
			}
			continue;
		}

		scale = max_abs / 127.0f;
		out->s[group] = scale;

		/* Quantize each activation */
		for (i = 0; i < group_len; i++)
		{
			scaled = in[offset + i] / scale;

			/*
			 * Round to nearest integer, halfway away from zero.
			 * Compatible with C89 (no round() dependency).
			 */
			if (scaled >= 0.0f)
			{
				quantized = (int)(scaled + 0.5f);
			}
			else
			{
				quantized = (int)(scaled - 0.5f);
			}

			/* Clamp to the Q8 range */
			if (quantized > 127)
			{
				quantized = 127;
			}
			else if (quantized < -127)
			{
				quantized = -127;
			}

			out->q[offset + i] = (signed char)quantized;
		}
	}
}

/*
 * Prepare an activation vector for matrix multiplication.
 *
 * FP32: no preparation required.
 * Q8_0: quantize the FP32 activation into the supplied buffer.
 */
void prepare_activation(const Transformer *t, const float *input,
			QuantizedTensor *quantized, size_t n, int group_size)
{
	if (t->model_format == MODEL_Q8_0)
	{
		quantize_q8(quantized, input, n, group_size);
	}
}

/* Map Q8_0 checkpoint tensors into the loaded payload.
   Convert FP32 normalization weights to native byte order,
   then map quantized matrices in the exporter's tensor order.
   Each quantized tensor contains int8 weights and FP32 scales. */
void memory_map_weights_q8(TransformerWeights *weights, Config *config,
			   void *data, size_t data_size, int shared_weights,
			   int group_size)
{
	unsigned char *ptr;
	size_t offset;
	size_t n;
	size_t bytes;
	size_t i;
	unsigned char *raw;
	size_t elements;
	size_t dim;
	size_t kv_dim;
	size_t layers;
	size_t attention_elements_per_layer;
	size_t kv_elements_per_layer;
	size_t layer;
	size_t hidden_dim;
	size_t ffn_elements_per_layer;

	/* Validate arguments before accessing configuration */
	if (weights == NULL || config == NULL || data == NULL)
	{
		fprintf(stderr, "Invalid arguments to Q8 weight mapper\n");
		exit(EXIT_FAILURE);
	}

	/* Validate model dimensions */
	if (config->dim <= 0 || config->hidden_dim <= 0 ||
	    config->n_layers <= 0)
	{
		fprintf(stderr, "Invalid Q8 model dimensions\n");
		exit(EXIT_FAILURE);
	}

	/* Convert validated dimensions to size_t */
	dim = (size_t)config->dim;
	hidden_dim = (size_t)config->hidden_dim;
	layers = (size_t)config->n_layers;

	/* Check for multiplication overflow */
	if (hidden_dim > (size_t)-1 / dim)
	{
		fprintf(stderr, "Q8 FFN tensor size overflow\n");
		exit(EXIT_FAILURE);
	}

	ffn_elements_per_layer = hidden_dim * dim;

	ptr = (unsigned char *)data;
	offset = 0;

	/* Attention RMSNorm weights: n_layers * dim */
	if ((size_t)config->n_layers > (size_t)-1 / (size_t)config->dim)
	{
		fprintf(stderr, "Q8 RMSNorm size overflow\n");
		exit(EXIT_FAILURE);
	}

	n = layers * dim;
	if (n > (size_t)-1 / sizeof(float))
	{
		fprintf(stderr, "Q8 RMSNorm byte size overflow\n");
		exit(EXIT_FAILURE);
	}

	bytes = n * sizeof(float);
	if (offset > data_size || bytes > data_size - offset)
	{
		fprintf(stderr, "Q8 checkpoint truncated (attention RMSNorm)\n");
		exit(EXIT_FAILURE);
	}

	/* Allocate per-layer Q8 attention tensor descriptors */
	if (layers > (size_t)-1 / sizeof(QuantizedTensor))
	{
		fprintf(stderr, "Q8 descriptor allocation size overflow\n");
		exit(EXIT_FAILURE);
	}

	weights->q_wq = calloc(layers, sizeof(QuantizedTensor));
	weights->q_wk = calloc(layers, sizeof(QuantizedTensor));
	weights->q_wv = calloc(layers, sizeof(QuantizedTensor));
	weights->q_wo = calloc(layers, sizeof(QuantizedTensor));

	/* Allocate per-layer Q8 feed-forward descriptors */
	weights->q_w1 = calloc(layers, sizeof(QuantizedTensor));
	weights->q_w2 = calloc(layers, sizeof(QuantizedTensor));
	weights->q_w3 = calloc(layers, sizeof(QuantizedTensor));

	if (weights->q_wq == NULL || weights->q_wk == NULL ||
	    weights->q_wv == NULL || weights->q_wo == NULL ||
	    weights->q_w1 == NULL || weights->q_w2 == NULL ||
	    weights->q_w3 == NULL)
	{
		fprintf(stderr, "Failed to allocate Q8 tensor descriptors\n");
		exit(EXIT_FAILURE);
	}

	weights->rms_att_weight = (float *)(ptr + offset);

	/* Convert little-endian FP32 values to native floats */
	raw = ptr + offset;
	for (i = 0; i < n; i++)
	{
		float value = le32_bytes_to_host_f32(raw + i * sizeof(float));
		memcpy(raw + i * sizeof(float), &value, sizeof(float));
	}

	offset += bytes;
	/* Feed-forward RMSNorm weights: n_layers * dim */
	if (offset > data_size || bytes > data_size - offset)
	{
		fprintf(stderr, "Q8 checkpoint truncated (FFN RMSNorm)\n");
		exit(EXIT_FAILURE);
	}

	weights->rms_ffn_weight = (float *)(ptr + offset);
	raw = ptr + offset;

	for (i = 0; i < n; i++)
	{
		float value = le32_bytes_to_host_f32(raw + i * sizeof(float));
		memcpy(raw + i * sizeof(float), &value, sizeof(float));
	}

	offset += bytes;

	/* Final RMSNorm weights: dim */
	n = dim;
	bytes = n * sizeof(float);
	if (offset > data_size || bytes > data_size - offset)
	{
		fprintf(stderr, "Q8 checkpoint truncated (final RMSNorm)\n");
		exit(EXIT_FAILURE);
	}

	weights->rms_final_weight = (float *)(ptr + offset);

	/* Normalize FP32 byte order in place; quantized int8 values
       require no conversion. */
	raw = ptr + offset;
	for (i = 0; i < n; i++)
	{
		float value = le32_bytes_to_host_f32(raw + i * sizeof(float));
		memcpy(raw + i * sizeof(float), &value, sizeof(float));
	}

	offset += bytes;

	/* Quantized token embedding table */
	if (config->vocab_size <= 0 || group_size <= 0)
	{
		fprintf(stderr, "Invalid Q8 embedding dimensions\n");
		exit(EXIT_FAILURE);
	}

	if ((size_t)config->vocab_size > (size_t)-1 / (size_t)config->dim)
	{
		fprintf(stderr, "Q8 embedding element count overflow\n");
		exit(EXIT_FAILURE);
	}

	elements = (size_t)config->vocab_size * dim;

	/* Validate attention configuration */
	if (config->dim <= 0 || config->n_layers <= 0 || config->n_heads <= 0 ||
	    config->n_kv_heads <= 0 || config->n_kv_heads > config->n_heads ||
	    config->dim % config->n_heads != 0 ||
	    config->n_heads % config->n_kv_heads != 0)
	{
		fprintf(stderr, "Invalid Q8 attention configuration\n");
		exit(EXIT_FAILURE);
	}

	/* Calculate KV dimension without intermediate overflow */
	kv_dim = (dim / (size_t)config->n_heads) * (size_t)config->n_kv_heads;

	/* Check tensor element counts */
	if (dim > (size_t)-1 / dim || kv_dim > (size_t)-1 / dim)
	{
		fprintf(stderr, "Q8 attention tensor size overflow\n");
		exit(EXIT_FAILURE);
	}

	attention_elements_per_layer = dim * dim;
	kv_elements_per_layer = kv_dim * dim;

	if (!map_q8_tensor(&weights->q_token_embedding_table, ptr, data_size,
			   &offset, elements, (size_t)group_size))
	{
		fprintf(stderr, "Failed to map Q8 embedding table\n");
		exit(EXIT_FAILURE);
	}

	/* ---------------------------------------------------------
	 * Quantized attention projection matrices
	 *
	 * Export order, with all layers:
	 *   WQ, WK, WV, WO
	 *
	 * Each layer is stored as:
	 *   [int8 weights][FP32 scales]
	 * --------------------------------------------------------- */

	/* Query projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_wq[layer], ptr, data_size,
				   &offset, attention_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 WQ layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* Key projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_wk[layer], ptr, data_size,
				   &offset, kv_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 WK layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* Value projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_wv[layer], ptr, data_size,
				   &offset, kv_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 WV layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* Output projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_wo[layer], ptr, data_size,
				   &offset, attention_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 WO layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* ---------------------------------------------------------
	 * Quantized feed-forward projection matrices
	 *
	 * Export order with all layers:
	 *   W1, W2, W3
	 *
	 * Each layer is stored as:
	 *   [int8 weights][FP32 scales]
	 * --------------------------------------------------------- */

	/* W1 projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_w1[layer], ptr, data_size,
				   &offset, ffn_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 W1 layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* W2 projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_w2[layer], ptr, data_size,
				   &offset, ffn_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 W2 layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* W3 projections */
	for (layer = 0; layer < layers; layer++)
	{
		if (!map_q8_tensor(&weights->q_w3[layer], ptr, data_size,
				   &offset, ffn_elements_per_layer,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 W3 layer %lu\n",
				(unsigned long)layer);
			exit(EXIT_FAILURE);
		}
	}

	/* ---------------------------------------------------------
	 * Quantized classifier weights
	 * --------------------------------------------------------- */
	if (shared_weights)
	{
		/* Classifier shares the token embedding weights */
		weights->q_wcls = weights->q_token_embedding_table;
	}
	else
	{
		/* Separate quantized classifier tensor */
		if (!map_q8_tensor(&weights->q_wcls, ptr, data_size, &offset,
				   (size_t)config->vocab_size * dim,
				   (size_t)group_size))
		{
			fprintf(stderr, "Failed to map Q8 classifier weights\n");
			exit(EXIT_FAILURE);
		}
	}

	if (offset != data_size)
	{
		fprintf(
		    stderr,
		    "Q8 checkpoint size mismatch: mapped %lu of %lu bytes\n",
		    (unsigned long)offset, (unsigned long)data_size);
		exit(EXIT_FAILURE);
	}
}

/* Load a legacy FP32 checkpoint.
   Read the model configuration, allocate storage for the weights
   and convert little-endian FP32 values to native host format.
   Tensor pointers are then assigned within the allocated buffer. */
void read_checkpoint_fp32(FILE *file, Config *config,
	TransformerWeights *weights, void **data,
	ssize_t *file_size)
{
	int shared_weights;
	float *weights_ptr;
	long weight_bytes;
	long n_weights;
	long i;
	unsigned char raw[4];
	if (!file)
	{
		fprintf(stderr, "Couldn't open file!\n");
		exit(EXIT_FAILURE);
	}
	/* Read in the config header */
	if (!read_le_i32(file, &config->dim) ||
	    !read_le_i32(file, &config->hidden_dim) ||
	    !read_le_i32(file, &config->n_layers) ||
	    !read_le_i32(file, &config->n_heads) ||
	    !read_le_i32(file, &config->n_kv_heads) ||
	    !read_le_i32(file, &config->vocab_size) ||
	    !read_le_i32(file, &config->seq_len))
	{
		fprintf(stderr, "Failed to read FP32 checkpoint header\n");
		exit(EXIT_FAILURE);
	}
	/* Explicitly set group_size to 0 since it doesn't exist in the FP32 file */
	config->group_size = 0;
	/* Legacy llama2.c convention: negative vocab_size means
	   the classifier has separate weights from the embedding table. */
	shared_weights = config->vocab_size > 0 ? 1 : 0;
	if (config->vocab_size == INT_MIN || config->vocab_size == 0)
	{
		fprintf(stderr, "Invalid FP32 vocabulary size\n");
		exit(EXIT_FAILURE);
	}
	config->vocab_size = abs(config->vocab_size);
	/* Figure out the file size */
	if (fseek(file, 0L, SEEK_END) != 0)
	{
		fprintf(stderr, "Failed to seek to end of FP32 checkpoint\n");
		exit(EXIT_FAILURE);
	}
	*file_size = ftell(file);
	if (*file_size < 0)
	{
		fprintf(stderr, "Failed to determine FP32 checkpoint size\n");
		exit(EXIT_FAILURE);
	}

	weight_bytes = (long)(*file_size - FP32_HEADER_SIZE);
	if (weight_bytes < 0 || (weight_bytes % 4) != 0)
	{
		fprintf(stderr, "Bad checkpoint size\n");
		exit(EXIT_FAILURE);
	}

	/* Read and convert all FP32 weights into host-native storage. */
	n_weights = weight_bytes / 4;
	weights_ptr = (float *)malloc((size_t)n_weights * sizeof(float));
	if (weights_ptr == NULL)
	{
		fprintf(stderr, "FP32 weight allocation failed: %ld bytes\n", weight_bytes);
		exit(EXIT_FAILURE);
	}
	/* Return to the beginning of the FP32 weights after measuring file size. */
	if (fseek(file, FP32_HEADER_SIZE, SEEK_SET) != 0)
	{
		fprintf(stderr, "Failed to seek to FP32 checkpoint weights\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < n_weights; i++)
	{
		if (fread(raw, 1, 4, file) != 4)
		{
			fprintf(stderr, "Failed to read FP32 checkpoint weights\n");
			exit(EXIT_FAILURE);
		}
		weights_ptr[i] = le32_bytes_to_host_f32(raw);
	}

	*data = weights_ptr;
	/* Assign tensor pointers into the allocated weight buffer. */
	memory_map_weights(weights, config, weights_ptr, shared_weights);
}

/* Load a Q8_0 checkpoint.
   Validate the versioned header, read the quantized weight payload
   and map its tensors, converting FP32 scales as needed. */
void read_checkpoint_q8(FILE *file, Config *config, TransformerWeights *weights,
			void **data, ssize_t *file_size)
{
	int magic;
	int version;
	unsigned char shared_classifier;
	int shared_weights;
	long total_size;
	long payload_size;
	unsigned char *buffer;

	if (!file)
	{
		fprintf(stderr, "Couldn't open file!\n");
		exit(EXIT_FAILURE);
	}

	/* Read magic and version */
	if (!read_le_i32(file, &magic) || !read_le_i32(file, &version))
	{
		fprintf(stderr, "Invalid Q8 checkpoint header\n");
		exit(EXIT_FAILURE);
	}
	if ((unsigned long)magic != CHECKPOINT_MAGIC ||
	    version != CHECKPOINT_VERSION_Q8)
	{
		fprintf(stderr, "Unsupported Q8 checkpoint version\n");
		exit(EXIT_FAILURE);
	}
	/* Read in the config header */
	if (!read_le_i32(file, &config->dim) ||
	    !read_le_i32(file, &config->hidden_dim) ||
	    !read_le_i32(file, &config->n_layers) ||
	    !read_le_i32(file, &config->n_heads) ||
	    !read_le_i32(file, &config->n_kv_heads) ||
	    !read_le_i32(file, &config->vocab_size) ||
	    !read_le_i32(file, &config->seq_len))
	{
		fprintf(stderr, "Failed to read Q8 checkpoint configuration\n");
		exit(EXIT_FAILURE);
	}

	/* Shared classifier flag: one byte at offset 36
	   Unlike legacy FP32, Q8_0 explicitly stores classifier weight sharing. */
	if (fread(&shared_classifier, 1, 1, file) != 1)
	{
		fprintf(stderr, "Failed to read shared classifier flag\n");
		exit(EXIT_FAILURE);
	}

	shared_weights = shared_classifier ? 1 : 0;

	/* Group size immediately follows the shared-classifier flag */
	if (!read_le_i32(file, &config->group_size))
	{
		fprintf(stderr, "Failed to read Q8 group size\n");
		exit(EXIT_FAILURE);
	}

	/* Validate the group size */
	if (config->group_size <= 0)
	{
		fprintf(stderr, "Invalid Q8 quantization group size\n");
		exit(EXIT_FAILURE);
	}

	/* Skip remaining header padding */
	if (fseek(file, 256L, SEEK_SET) != 0)
	{
		fprintf(stderr, "Failed to seek to Q8 weight data\n");
		exit(EXIT_FAILURE);
	}

	/* Determine checkpoint size */
	if (fseek(file, 0L, SEEK_END) != 0)
	{
		fprintf(stderr, "Failed to seek to end of Q8 checkpoint\n");
		exit(EXIT_FAILURE);
	}

	total_size = ftell(file);
	if (total_size <= 256L)
	{
		fprintf(stderr, "Invalid Q8 checkpoint size\n");
		exit(EXIT_FAILURE);
	}

	/* Calculate payload size */
	payload_size = total_size - 256L;

	/* Ensure payload fits in size_t */
	if ((unsigned long)payload_size > (size_t)-1)
	{
		fprintf(stderr,
			"Q8 checkpoint too large for memory allocation\n");
		exit(EXIT_FAILURE);
	}

	/* Allocate exactly the required number of bytes */
	buffer = (unsigned char *)malloc((size_t)payload_size);
	if (buffer == NULL)
	{
		fprintf(stderr, "Failed to allocate memory for Q8 weights\n");
		exit(EXIT_FAILURE);
	}

	/* Return to beginning of weight data */
	if (fseek(file, 256L, SEEK_SET) != 0)
	{
		fprintf(stderr, "Failed to seek to Q8 weight data\n");
		free(buffer);
		exit(EXIT_FAILURE);
	}

	/* Read entire payload */
	if (fread(buffer, 1, (size_t)payload_size, file) !=
	    (size_t)payload_size)
	{
		fprintf(stderr, "Failed to read Q8 weight data\n");
		free(buffer);
		exit(EXIT_FAILURE);
	}

	/* Store allocation for later mapping and cleanup */
	*data = buffer;
	*file_size = (ssize_t)total_size;

	memory_map_weights_q8(weights, config, *data, (size_t)payload_size,
			      shared_weights, config->group_size);
}

/* Read the checkpoint. Detect the model format and dispatch
   format specific checkpoint reader */
void read_checkpoint(char *checkpoint, Config *config,
		     TransformerWeights *weights, void **data,
		     ssize_t *file_size, int *model_format)
{
	FILE *file;
	file = fopen(checkpoint, "rb");
	if (file == NULL)
	{
		fprintf(stderr, "Error opening checkpoint\n");
		perror(checkpoint);
		exit(EXIT_FAILURE);
	}

	*model_format = detect_model_format(file);
	/* Reset file position after reading the format identifier. */
	rewind(file);

	if (*model_format == MODEL_Q8_0)
	{
		fprintf(stderr, "Detected Q8_0 format...\n\n");
		read_checkpoint_q8(file, config, weights, data, file_size);
	}
	else if (*model_format == MODEL_FP32)
	{
		fprintf(stderr, "Detected FP32 format...\n\n");
		read_checkpoint_fp32(file, config, weights, data, file_size);
	}
	else
	{
		fprintf(stderr, "Unable to detect checkpoint format!\n\n");
		fclose(file);
		exit(EXIT_FAILURE);
	}
	fclose(file);
}

/* Builds the Transformer. 
   Reads in the checkpoint and allocates the memory */
void build_transformer(Transformer *t, char *checkpoint_path)
{
	/* Read in the Config and the Weights from the checkpoint */
	memset(&t->weights, 0, sizeof(t->weights));
	read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data,
			&t->file_size, &t->model_format);

	/* Allocate the RunState buffers */
	malloc_run_state(&t->state, &t->config);
	/* Allocate additional buffers for quantized inference */
	if (t->model_format == MODEL_Q8_0)
	{
		malloc_run_state_q8(&t->state, &t->config);
	}
}

/* Free memory used by Transformer */
void free_transformer(Transformer *t)
{
	/* Free Q8 attention tensor descriptors */
	free(t->weights.q_wq);
	free(t->weights.q_wk);
	free(t->weights.q_wv);
	free(t->weights.q_wo);
	/* Free Q8 feed-forward tensor descriptors */
	free(t->weights.q_w1);
	free(t->weights.q_w2);
	free(t->weights.q_w3);
	/* Free checkpoint weight data */
	free(t->data);
	/* Free runtime buffers */
	free_run_state(&t->state);
}

/* Perform Root Mean Square Normalization (RMSNorm) */
void rmsnorm(float *o, float *x, float *weight, int size)
{
	/* Calculate sum of squares */
	float ss = 0.0f;
	int j;
	for (j = 0; j < size; j++)
	{
		ss += x[j] * x[j];
	}
	ss /= size;
	ss += 1e-5f;
	ss = 1.0f / sqrt(ss);
	/* Normalize and scale */
	for (j = 0; j < size; j++)
	{
		o[j] = weight[j] * (ss * x[j]);
	}
}

/* Perform Softmax */
void softmax(float *x, int size)
{
	/* Find max value (for numerical stability) */
	float max_val = x[0];
	int i;
	float sum = 0.0f;
	for (i = 1; i < size; i++)
	{
		if (x[i] > max_val)
		{
			max_val = x[i];
		}
	}
	/* exp and sum */
	for (i = 0; i < size; i++)
	{
		x[i] = exp(x[i] - max_val);
		sum += x[i];
	}
	/* Normalize */
	for (i = 0; i < size; i++)
	{
		x[i] /= sum;
	}
}

/* FP32 matrix-vector multiplication (row-major weights). */
void matmul(float *xout, const float *x, const float *w, int n, int d)
{
	/* W (d,n) @ x (n,) -> xout (d,) */
	int i, j;
	for (i = 0; i < d; i++)
	{
		float val = 0.0f;
		for (j = 0; j < n; j++)
		{
			val += w[(size_t)i * (size_t)n + (size_t)j] * x[j];
		}
		xout[i] = val;
	}
}

/*
  Matrix multiplication using grouped Q8_0 weights and activations.
 
  xout = W * x
 
  W is an (n x d) matrix stored as grouped Q8_0 values.
  x is a quantized vector containing d elements.
 
  Each group contains group_size signed 8-bit values and
  one FP32 scaling factor.
 
  Both input tensors must already be quantized. */
void matmul_q8(float *xout, const QuantizedTensor *x, const QuantizedTensor *w,
	       size_t n, size_t d, size_t group_size)
{
	size_t i;
	size_t j;
	size_t k;
	size_t row_offset;
	size_t weight_group;
	size_t activation_group;
	size_t weight_index;
	size_t weight_remaining;
	size_t activation_remaining;
	size_t count;

	long group_sum;
	float value;

	if (xout == NULL || x == NULL || w == NULL || x->q == NULL ||
	    x->s == NULL || w->q == NULL || w->s == NULL)
	{
		fprintf(stderr, "Invalid Q8 matrix multiplication buffers\n");
		exit(EXIT_FAILURE);
	}

	if (n == 0 || d == 0 || group_size == 0)
	{
		fprintf(stderr,
			"Invalid Q8 matrix multiplication dimensions\n");
		exit(EXIT_FAILURE);
	}

	/* Check matrix element count for overflow */
	if (n > (size_t)-1 / d)
	{
		fprintf(stderr, "Q8 matrix multiplication size overflow\n");
		exit(EXIT_FAILURE);
	}

	/* Ensure a group cannot overflow the signed long accumulator */
	if (group_size > (size_t)LONG_MAX / (127L * 127L))
	{
		fprintf(stderr, "Q8 group size exceeds accumulator capacity\n");
		exit(EXIT_FAILURE);
	}

	/* Stored Q8 weights must contain complete flattened groups */
	if ((n * d) % group_size != 0)
	{
		fprintf(stderr, "Invalid Q8 weight tensor dimensions\n");
		exit(EXIT_FAILURE);
	}

	/* Weight groups follow the flattened matrix, while activation
       groups follow the input vector. Their boundaries may differ
       within a matrix row, so each dot-product segment must stop
       at the next boundary of either group. */
	for (i = 0; i < n; i++)
	{
		row_offset = i * d;
		value = 0.0f;
		j = 0;

		while (j < d)
		{
			weight_index = row_offset + j;
			weight_group = weight_index / group_size;
			activation_group = j / group_size;

			weight_remaining =
			    group_size - (weight_index % group_size);

			activation_remaining = group_size - (j % group_size);

			count = weight_remaining;
			if (count > activation_remaining)
			{
				count = activation_remaining;
			}

			if (count > d - j)
			{
				count = d - j;
			}

			group_sum = 0;

			for (k = 0; k < count; k++)
			{
				group_sum += (long)w->q[weight_index + k] *
					     (long)x->q[j + k];
			}

			value += (float)group_sum * w->s[weight_group] *
				 x->s[activation_group];

			j += count;
		}

		xout[i] = value;
	}
}

/*
 * Matrix projection using FP32 or Q8_0 weights.
 *
 * Q8 activations must already have been prepared.
 */
void project(float *output, const float *input,
	     const QuantizedTensor *quantized_input, const float *fp32_weights,
	     const QuantizedTensor *q8_weights, int input_dim, int output_dim,
	     int model_format, int group_size)
{
	if (input_dim <= 0 || output_dim <= 0)
	{
		fprintf(stderr, "Invalid matrix projection dimensions\n");
		exit(EXIT_FAILURE);
	}

	if (model_format == MODEL_Q8_0)
	{
		matmul_q8(output, quantized_input, q8_weights,
			  (size_t)output_dim, (size_t)input_dim, group_size);
	}
	else
	{
		matmul(output, input, fp32_weights, input_dim, output_dim);
	}
}

/*
  Retrieve a token embedding into the FP32 activation buffer.
 
  FP32 checkpoints: copy the embedding directly.
  Q8_0 checkpoints: dequantize the embedding using group scales. */
void embedding_lookup(Transformer *t, int token, int group_size)
{
	Config *p;
	TransformerWeights *w;
	RunState *s;
	size_t dim;
	size_t offset;
	size_t i;
	size_t group;
	const float *embedding;

	p = &t->config;
	w = &t->weights;
	s = &t->state;

	if (token < 0 || token >= p->vocab_size || p->dim <= 0)
	{
		fprintf(stderr, "Invalid token index or embedding dimension\n");
		exit(EXIT_FAILURE);
	}

	dim = (size_t)p->dim;
	/* Prevent overflow in the embedding offset calculation. */
	if ((size_t)token > (size_t)-1 / dim)
	{
		fprintf(stderr, "Embedding offset overflow\n");
		exit(EXIT_FAILURE);
	}

	offset = (size_t)token * dim;

	if (t->model_format == MODEL_Q8_0)
	{
		if (group_size <= 0)
		{
			fprintf(stderr, "Invalid Q8 embedding group size\n");
			exit(EXIT_FAILURE);
		}

		for (i = 0; i < dim; i++)
		{
			group = (offset + i) / group_size;

			s->x[i] =
			    (float)w->q_token_embedding_table.q[offset + i] *
			    w->q_token_embedding_table.s[group];
		}
	}
	else
	{
		embedding = w->token_embedding_table + offset;
		memcpy(s->x, embedding, dim * sizeof(*s->x));
	}
}

/* Run one token through the transformer.
   Applies token embeddings, attention, feed-forward layers,
   residual connections and final classifier projection.
   Returns the vocabulary logits for the current position. */
float *forward(Transformer *transformer, int token, int pos)
{
	Config *p;
	TransformerWeights *w;
	RunState *s;
	float *x;
	int dim, kv_dim, kv_mul, hidden_dim, head_size;
	int l, i, h, t, v;
	int head_dim, rotn;
	float freq, val, fcr, fci;
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

	const float *fp32_wq;
	const float *fp32_wk;
	const float *fp32_wv;
	const float *fp32_wo;
	const float *fp32_w1;
	const float *fp32_w3;
	const float *fp32_w2;
	const float *fp32_wcls;

	const QuantizedTensor *q8_wq;
	const QuantizedTensor *q8_wk;
	const QuantizedTensor *q8_wv;
	const QuantizedTensor *q8_wo;
	const QuantizedTensor *q8_w1;
	const QuantizedTensor *q8_w2;
	const QuantizedTensor *q8_w3;
	const QuantizedTensor *q8_wcls;

	size_t loff;

	p = &transformer->config;
	w = &transformer->weights;
	s = &transformer->state;
	x = s->x;

	dim = p->dim;
	kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
	kv_mul = p->n_heads / p->n_kv_heads;
	hidden_dim = p->hidden_dim;
	head_size = dim / p->n_heads;

	embedding_lookup(transformer, token, p->group_size);

	/* Feed Forward all of the model layers.. */
	for (l = 0; l < p->n_layers; l++)
	{
		if (transformer->model_format == MODEL_Q8_0)
		{
			fp32_wq = NULL;
			fp32_wk = NULL;
			fp32_wv = NULL;
			fp32_wo = NULL;
			fp32_w1 = NULL;
			fp32_w2 = NULL;
			fp32_w3 = NULL;

			q8_w1 = &w->q_w1[l];
			q8_w2 = &w->q_w2[l];
			q8_w3 = &w->q_w3[l];
			q8_wq = &w->q_wq[l];
			q8_wk = &w->q_wk[l];
			q8_wv = &w->q_wv[l];
			q8_wo = &w->q_wo[l];
		}
		else /* FP32 format*/
		{
			fp32_wq = w->wq + (size_t)l * dim * dim;
			fp32_wk = w->wk + (size_t)l * dim * kv_dim;
			fp32_wv = w->wv + (size_t)l * dim * kv_dim;
			fp32_wo = w->wo + (size_t)l * dim * dim;
			fp32_w1 = w->w1 + (size_t)l * dim * hidden_dim;
			fp32_w2 = w->w2 + (size_t)l * dim * hidden_dim;
			fp32_w3 = w->w3 + (size_t)l * dim * hidden_dim;

			q8_w1 = NULL;
			q8_w2 = NULL;
			q8_w3 = NULL;
			q8_wo = NULL;
			q8_wq = NULL;
			q8_wk = NULL;
			q8_wv = NULL;
		}

		/* Attention RMSNorm normalisation */
		rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

		prepare_activation(transformer, s->xb, &s->q_x, (size_t)dim,
				   p->group_size);

		project(s->q, s->xb, &s->q_x, fp32_wq, q8_wq, dim, dim,
			transformer->model_format, p->group_size);

		project(s->k, s->xb, &s->q_x, fp32_wk, q8_wk, dim, kv_dim,
			transformer->model_format, p->group_size);

		project(s->v, s->xb, &s->q_x, fp32_wv, q8_wv, dim, kv_dim,
			transformer->model_format, p->group_size);

		/* RoPE relative positional encoding: complex-valued rotate q and k in each head */
		for (i = 0; i < dim; i += 2)
		{
			head_dim = i % head_size;
			freq = 1.0f / pow(10000.0, head_dim / (float)head_size);
			val = pos * freq;
			fcr = cos(val);
			fci = sin(val);
			rotn = i < kv_dim ? 2 : 1;

			for (v = 0; v < rotn; v++)
			{
				vec = v == 0 ? s->q : s->k;
				v0 = vec[i];
				v1 = vec[i + 1];
				vec[i] = v0 * fcr - v1 * fci;
				vec[i + 1] = v0 * fci + v1 * fcr;
			}
		}

		loff = (size_t)l * (size_t)p->seq_len * (size_t)kv_dim;
		key_cache_row = s->key_cache + loff + pos * kv_dim;
		value_cache_row = s->value_cache + loff + pos * kv_dim;
		memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
		memcpy(value_cache_row, s->v,
		       kv_dim * sizeof(*value_cache_row));

		/* Multihead attention. iterate over all heads */
		for (h = 0; h < p->n_heads; h++)
		{
			/* Get the query vector for this head */
			q = s->q + h * head_size;
			/* Attention scores for this head */
			att = s->att + h * p->seq_len;
			/* Iterate over all timesteps, including the current one */
			for (t = 0; t <= pos; t++)
			{
				/* Get the key vector for this head and at this timestep */
				k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				/* Calculate the attention score as the dot product of Q and K */
				score = 0.0f;
				for (i = 0; i < head_size; i++)
				{
					score += q[i] * k[i];
				}
				score /= sqrt((float)head_size);
				/* Save the score to the attention buffer */
				att[t] = score;
			}

			/* Softmax the scores to get attention weights, from 0..pos inclusively */
			softmax(att, pos + 1);
			/* Weighted sum of the values, store back into xb */
			xb = s->xb + h * head_size;
			memset(xb, 0, head_size * sizeof(float));

			for (t = 0; t <= pos; t++)
			{
				/* Get the value vector for this head and at this timestep */
				vptr = s->value_cache + loff + t * kv_dim +
				       (h / kv_mul) * head_size;
				/* Get the attention weight for this timestep */
				a = att[t];
				/* Accumulate the weighted value into xb */
				for (i = 0; i < head_size; i++)
				{
					xb[i] += a * vptr[i];
				}
			}
		}

		prepare_activation(transformer, s->xb, &s->q_x, (size_t)dim,
				   p->group_size);

		project(s->xb2, s->xb, &s->q_x, fp32_wo, q8_wo, dim, dim,
			transformer->model_format, p->group_size);

		/* Residual connection back into x */
		for (i = 0; i < dim; i++)
		{
			x[i] += s->xb2[i];
		}
		/* Feed Forward Network RMSNorm normalisation */
		rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

		/* Prepare the normalized activation once for W1 and W3. */
		prepare_activation(transformer, s->xb, &s->q_x, (size_t)dim,
				   p->group_size);

		project(s->hb, s->xb, &s->q_x, fp32_w1, q8_w1, dim, hidden_dim,
			transformer->model_format, p->group_size);

		project(s->hb2, s->xb, &s->q_x, fp32_w3, q8_w3, dim, hidden_dim,
			transformer->model_format, p->group_size);

		/* SwiGLU non-linearity */
		for (i = 0; i < hidden_dim; i++)
		{
			val = s->hb[i];
			val *= (1.0f / (1.0f + exp(-val)));
			val *= s->hb2[i];
			s->hb[i] = val;
		}

		/* Prepare the hidden activation for the W2 projection. */
		prepare_activation(transformer, s->hb, &s->q_hb,
				   (size_t)hidden_dim, p->group_size);

		project(s->xb, s->hb, &s->q_hb, fp32_w2, q8_w2, hidden_dim, dim,
			transformer->model_format, p->group_size);

		/* Residual connection */
		for (i = 0; i < dim; i++)
		{
			x[i] += s->xb[i];
		}
	}

	/* Final RMSNorm normalization */
	rmsnorm(x, x, w->rms_final_weight, dim);

	/* Select the classifier weights. */
	if (transformer->model_format == MODEL_Q8_0)
	{
		fp32_wcls = NULL;
		q8_wcls = &w->q_wcls;
	}
	else
	{
		fp32_wcls = w->wcls;
		q8_wcls = NULL;
	}

	/* Prepare the final activation for the classifier. */
	prepare_activation(transformer, x, &s->q_x, (size_t)dim, p->group_size);

	/* Calculate vocabulary logits. */
	project(s->logits, x, &s->q_x, fp32_wcls, q8_wcls, dim, p->vocab_size,
		transformer->model_format, p->group_size);

	return s->logits;
}

int compare_tokens(const void *a, const void *b)
{
	return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

/* Load the Tokenizer file and build it. Performs endianness 
   byte swapping if needed */
void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{
	int i;
	int len;
	FILE *file;
	if (vocab_size <= 0 || (size_t)vocab_size > (size_t)-1 / sizeof(char *) ||
       (size_t)vocab_size > (size_t)-1 / sizeof(float))
	{
		fprintf(stderr, "Invalid tokenizer vocabulary size\n");
		exit(EXIT_FAILURE);
	}
	t->vocab_size = vocab_size;
	/* malloc space to hold the scores and the strings */
	t->vocab = (char **)calloc((size_t)vocab_size, sizeof(char *));
	t->vocab_scores = (float *)malloc((size_t)vocab_size * sizeof(float));
	if (t->vocab == NULL || t->vocab_scores == NULL)
	{
		fprintf(stderr, "Failed to allocate tokenizer vocabulary\n");
		exit(EXIT_FAILURE);
	}
	t->sorted_vocab = NULL; /* Initialized lazily */
	for (i = 0; i < 256; i++)
	{
		t->byte_pieces[i * 2] = (unsigned char)i;
		t->byte_pieces[i * 2 + 1] = '\0';
	}
	/* Read in the file */
	file = fopen(tokenizer_path, "rb");
	if (!file)
	{
		fprintf(stderr, "Couldn't load %s\n", tokenizer_path);
		exit(EXIT_FAILURE);
	}
	if (!read_le_u32(file, &t->max_token_length))
	{
		fprintf(stderr, "Failed to read tokenizer header\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < vocab_size; i++)
	{
		if (!read_le_f32(file, t->vocab_scores + i))
		{
			fprintf(stderr, "Failed to read tokenizer score at entry %d\n", i);
			exit(EXIT_FAILURE);
		}
		if (!read_le_i32(file, &len))
		{
			fprintf(stderr, "Failed to read tokenizer length at entry %d\n", i);
			exit(EXIT_FAILURE);
		}
		if (len <= 0 || (size_t)len > (size_t)-1 - 1)
		{
			fprintf(stderr, "Invalid tokenizer length at entry %d\n", i);
			exit(EXIT_FAILURE);
		}
		t->vocab[i] = (char *)malloc((size_t)len + 1);
		if (t->vocab[i] == NULL)
		{
			fprintf(stderr, "Failed to allocate tokenizer entry %d\n", i);
			exit(EXIT_FAILURE);
		}
		if (fread(t->vocab[i], 1, (size_t)len, file) != (size_t)len)
		{
			fprintf(stderr, "Failed to read tokenizer entry %d\n", i);
			exit(EXIT_FAILURE);
		}
		/* Terminate string */
		t->vocab[i][len] = '\0';
	}
	fclose(file);
}

/* Free memory used by Tokenizer */
void free_tokenizer(Tokenizer *t)
{
	int i;
	for (i = 0; i < t->vocab_size; i++)
	{
		free(t->vocab[i]);
	}
	free(t->vocab);
	free(t->vocab_scores);
	free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
	char *piece;
	unsigned char byte_val;
	unsigned int parsed_byte;

	if (token < 0 || token >= t->vocab_size)
	{
		fprintf(stderr, "Invalid token index in decode: %d\n", token);
		exit(EXIT_FAILURE);
	}

	piece = t->vocab[token];

	/* Following BOS (1), strip one leading space. */
	if (prev_token == 1 && piece[0] == ' ')
	{
		piece++;
	}

	/*  Some tokens designate raw bytes, and look like e.g. '<0x01>'
	    Parse this and convert and return the actual byte */
	if (strlen(piece) == 6 &&
		piece[0] == '<' &&
		piece[1] == '0' &&
		piece[2] == 'x' &&
		piece[5] == '>' &&
		sscanf(piece, "<0x%2X>", &parsed_byte) == 1 &&
		parsed_byte <= 255)
	{
		byte_val = (unsigned char)parsed_byte;
		piece = (char *)t->byte_pieces + byte_val * 2;
	}

	return piece;
}

void safe_printf(char *piece)
{
	/* Piece might be a raw byte token, and we only want to print printable
	   chars or whitespace because some of the other bytes can be various 
	   control codes, backspace, etc. */
	if (piece == NULL)
	{
		return;
	}
	if (piece[0] == '\0')
	{
		return;
	}
	if (piece[1] == '\0')
	{
		unsigned char byte_val = piece[0];
		if (!(isprint(byte_val) || isspace(byte_val)))
		{
			return; /* Bad byte, do not print it */
		}
	}
	printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
	/* Find the perfect match for str in vocab, return its 
	   index or -1 if not found */
	TokenIndex tok;
	TokenIndex *res;
	tok.str = str; /* Acts as the key for search */
	tok.id = 0;
	res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex),
		      compare_tokens);
	return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens,
	    int *n_tokens)
{
	/* Encode the string text (input) into an upper-bound pre-allocated
	   tokens[] array bos != 0 means prepend the BOS token (=1), eos != 0 
	   means append the EOS token (=2) */
	int i;
	char *str_buffer;
	size_t str_len = 0;
	size_t buffer_size;
	char *c;
	if (text == NULL)
	{
		fprintf(stderr, "Cannot encode NULL text\n");
		exit(EXIT_FAILURE);
	}

	if (t->sorted_vocab == NULL)
	{
		if (t->vocab_size <= 0 || (size_t)t->vocab_size > (size_t)-1 / sizeof(TokenIndex))
		{
			fprintf(stderr, "Invalid tokenizer vocabulary size\n");
			exit(EXIT_FAILURE);
		}

		t->sorted_vocab = malloc((size_t)t->vocab_size * sizeof(TokenIndex));

		if (t->sorted_vocab == NULL)
		{
			fprintf(stderr, "Failed to allocate sorted vocabulary\n");
			exit(EXIT_FAILURE);
		}
		/* Lazily malloc and sort the vocabulary */
		/*t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));*/
		for (i = 0; i < t->vocab_size; i++)
		{
			t->sorted_vocab[i].str = t->vocab[i];
			t->sorted_vocab[i].id = i;
		}
		qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
	}

	/* Create a temporary buffer that will store merge candidates of always
	   two consecutive tokens 2 for concat, +1 for null terminator +2 for UTF8 
	   (in case max_token_length is 1) */
	if ((size_t)t->max_token_length > ((size_t)-1 - 3) / 2)
	{
		fprintf(stderr, "Tokenizer buffer size overflow\n");
		exit(EXIT_FAILURE);
	}

	buffer_size = (size_t)t->max_token_length * 2 + 3;
	if (buffer_size < 5)
	{
		buffer_size = 5;
	}

	str_buffer = (char *)malloc(buffer_size);
	if (str_buffer == NULL)
	{
		fprintf(stderr, "Failed to allocate tokenizer buffer\n");
		exit(EXIT_FAILURE);
	}

	/* Start at 0 tokens */
	*n_tokens = 0;

	/* Add optional BOS (=1) token, if desired */
	if (bos) 
	{
		tokens[(*n_tokens)++] = 1;
	}
	/* Add_dummy_prefix is true by default so prepend a dummy prefix token to 
	   the input string, but only if text != "" */
	if (text[0] != '\0')
	{
		int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
		if (dummy_prefix < 0)
		{
			fprintf(stderr, "Tokenizer vocabulary missing dummy prefix\n");
			exit(EXIT_FAILURE);
		}
		tokens[(*n_tokens)++] = dummy_prefix;
	}

	/* Process the raw (UTF-8) byte sequence of the input string */
	{
		for (c = text; *c != '\0'; c++)
		{
			int id;

			/* Reset buffer if the current byte is ASCII or a leading byte 
			   0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest 
			   0x80 is 10000000 in UTF-8, all continuation bytes start with "10" in first two bits */
			if (((unsigned char)*c & 0xC0) != 0x80)
			{
				/* This byte must be either a leading byte (11...) or an ASCII char (0x...)
				   => reset our location, as we are starting a new UTF-8 codepoint */
				str_len = 0;
			}

			/* Append the current byte to the buffer */
			str_buffer[str_len++] = *c;
			str_buffer[str_len] = '\0';

			/* While the next character is a continuation byte, continue appending 
			   but if there are too many of them, just stop to avoid overruning str_buffer size. */
			if (((unsigned char)*(c + 1) & 0xC0) == 0x80 && str_len < 4)
			{
				continue;
			}

			/* Ok c+1 is not a continuation byte, so we have read in a full codepoint */
			id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

			if (id != -1)
			{
				/* We found this codepoint in vocab, add it as a token */
				tokens[(*n_tokens)++] = id;
			}
			else
			{
				/* byte_fallback encoding: just encode each byte as a token
				   +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
				   so the individual bytes only start at index 3 */
				int i;
				for (i = 0; i < (int)str_len; i++)
				{
					tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
				}
			}
			str_len = 0; /* Protect against a sequence of stray UTF8 continuation bytes */
		}
	}

	/* Merge the best consecutive pair each iteration, according the scores in vocab_scores */
	while (1)
	{
		float best_score = -1e10;
		int best_id = -1;
		int best_idx = -1;
		int i;
		int id;

		for (i = 0; i < (*n_tokens - 1); i++)
		{
			/* Check if we can merge the pair (tokens[i], tokens[i+1]) */
			sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
			id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
			if (id != -1 && t->vocab_scores[id] > best_score)
			{
				/* This merge pair exists in vocab! record its score and position */
				best_score = t->vocab_scores[id];
				best_id = id;
				best_idx = i;
			}
		}

		if (best_idx == -1)
		{
			break; /* We could not find any more pairs to merge, so we are done */
		}

		/* Merge the consecutive pair (best_idx, best_idx+1) into new token best_id */
		tokens[best_idx] = best_id;
		/* Delete token at position best_idx+1, shift the entire sequence back 1 */
		for (i = best_idx + 1; i < (*n_tokens - 1); i++)
		{
			tokens[i] = tokens[i + 1];
		}
		(*n_tokens)--; /* Token length decreased */
	}

	/* Add optional EOS (=2) token, if desired */
	if (eos) 
	{
		tokens[(*n_tokens)++] = 2;
	}
	free(str_buffer);
}

int sample_argmax(float *probabilities, int n)
{
	/* Return the index that has the highest probability */
	int max_i = 0;
	float max_p = probabilities[0];
	int i;
	for (i = 1; i < n; i++)
	{
		if (probabilities[i] > max_p)
		{
			max_i = i;
			max_p = probabilities[i];
		}
	}
	return max_i;
}

int sample_mult(float *probabilities, int n, float coin)
{
	/* Sample index from probabilities (they must sum to 1!)
	   coin is a random number in [0, 1), usually from random_f32() */
	float cdf = 0.0f;
	int i;
	for (i = 0; i < n; i++)
	{
		cdf += probabilities[i];
		if (coin < cdf)
		{
			return i;
		}
	}
	return n - 1; /* In case of rounding errors */
}

int compare(const void *a, const void *b)
{
	ProbIndex *a_ = (ProbIndex *)a;
	ProbIndex *b_ = (ProbIndex *)b;
	if (a_->prob > b_->prob)
		return -1;
	if (a_->prob < b_->prob)
		return 1;
	return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex,
		float coin)
{
	/* top-p sampling (or "nucleus sampling") samples from the smallest set
	   of tokens that exceed probability topp. This way we never sample tokens
	   that have very low probabilities and are less likely to go "off the
	   rails". coin is a random number in [0, 1), usually from random_f32() */
	int n0 = 0;
	/* Quicksort indices in descending order of probabilities values
	   smaller than (1 - topp) / (n - 1) cannot be part of the result so for
	   efficiency we crop these out as candidates before sorting */
	float cutoff;
	float cumulative_prob;
	int last_idx;
	float r;
	float cdf;
	int i;
    if (n <= 1)
	{
		return 0;
	}
	cutoff = (1.0f - topp) / (n - 1);
	cdf = 0.0f;
	cumulative_prob = 0.0f;
	for (i = 0; i < n; i++)
	{
		if (probabilities[i] >= cutoff)
		{
			probindex[n0].index = i;
			probindex[n0].prob = probabilities[i];
			n0++;
		}
	}

	if (n0 == 0)
	{
		fprintf(stderr, "No candidates available for top-p sampling\n");
		exit(EXIT_FAILURE);
	}

	qsort(probindex, n0, sizeof(ProbIndex), compare);

	/* Truncate the list where cumulative probability exceeds topp */
	last_idx = n0 - 1; /* In case of rounding errors consider all elements */
	for (i = 0; i < n0; i++)
	{
		cumulative_prob += probindex[i].prob;
		if (cumulative_prob > topp)
		{
			last_idx = i;
			break; /* We have exceeded topp by including last_idx */
		}
	}

	/* Sample from the truncated list */
	r = coin * cumulative_prob;
	for (i = 0; i <= last_idx; i++)
	{
		cdf += probindex[i].prob;
		if (r < cdf)
		{
			return probindex[i].index;
		}
	}
	return probindex[last_idx].index; /* In case of rounding errors */
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature,
		   float topp, unsigned long rng_seed)
{
	sampler->vocab_size = vocab_size;
	sampler->temperature = temperature;
	sampler->topp = topp;
	sampler->rng_state = rng_seed;
	/* Buffer only used with nucleus sampling; may not need but it is small */
	sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long *state)
{
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;
	return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long *state)
{ /*  random float32 in [0,1) */
	return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits)
{
	/* Sample the token given the logits and some hyperparameters */
	int next;
	if (sampler->temperature == 0.0f)
	{
		/* Greedy argmax sampling: Select the token with the highest logit. */
		next = sample_argmax(logits, sampler->vocab_size);
	}
	else
	{
		float coin;
		/* Apply the temperature to the logits */
		int q;
		for (q = 0; q < sampler->vocab_size; q++)
		{
			logits[q] /= sampler->temperature;
		}
		/* Apply softmax to the logits to get the probabilities for next token */
		softmax(logits, sampler->vocab_size);
		/* Flip a (float) coin (this is our source of entropy for sampling) */
		coin = random_f32(&sampler->rng_state);
		/* We sample from this distribution to get the next token */
		if (sampler->topp <= 0 || sampler->topp >= 1)
		{
			/* Simply sample from the predicted probability distribution */
			next = sample_mult(logits, sampler->vocab_size, coin);
		}
		else
		{
			/*  top-p (nucleus) sampling, clamping the least likely tokens to zero */
			next = sample_topp(logits, sampler->vocab_size,
					   sampler->topp, sampler->probindex,
					   coin);
		}
	}
	return next;
}

long time_in_ms(void) { return (long)((clock() * 1000) / CLOCKS_PER_SEC); }

/* The main generation loop */
void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
	      char *prompt, int steps)
{
	char *empty_prompt = "";
	int num_prompt_tokens = 0;
	int *prompt_tokens;
	long start = 0; /* Used to time our code, only initialized after first iteration */
	int next;
	int token;
	int pos = 0;
	size_t prompt_len;
    size_t token_capacity;

	if (prompt == NULL)
	{
		prompt = empty_prompt;
	}

	prompt_len = strlen(prompt);

	if (prompt_len > (size_t)-1 / sizeof(int) - 3)
	{
		fprintf(stderr, "Prompt too large\n");
		exit(EXIT_FAILURE);
	}
	/* Encode the (string) prompt into tokens sequence 
	   Worst case: one token per input byte, plus BOS, EOS and dummy prefix. */
	token_capacity = prompt_len + 3;
	prompt_tokens = (int *)malloc(token_capacity * sizeof(int));
	if (prompt_tokens == NULL)
	{
		fprintf(stderr, "Failed to allocate prompt tokens\n");
		exit(EXIT_FAILURE);
	}
	encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
	if (num_prompt_tokens < 1)
	{
		fprintf(stderr, "Something is wrong, expected at least 1 prompt token\n");
		exit(EXIT_FAILURE);
	}

	/* Start the main loop */
	token = prompt_tokens[0]; /* Kick off with the first token in the prompt */
	if (steps > transformer->config.seq_len)
	{
		steps = transformer->config.seq_len;
	}
	while (pos < steps)
	{
		char *piece;
		float *logits;

		/* Forward the transformer to get logits for the next token */
		logits = forward(transformer, token, pos);

		/* Advance the state machine */
		if (pos < num_prompt_tokens - 1)
		{
			/* If we are still processing the input prompt, force the next prompt token */
			next = prompt_tokens[pos + 1];
		}
		else
		{
			/* Otherwise sample the next token from the logits */
			next = sample(sampler, logits);
		}
		pos++;

		/* Data-dependent terminating condition: the BOS (=1) token delimits sequences */
		if (next == 1)
		{
			break;
		}

		/* Print the token as string, decode it with the Tokenizer object */
		piece = decode(tokenizer, token, next);
		safe_printf(piece);
		fflush(stdout);
		token = next;

		/* Init the timer here because the first iteration can be slower */
		if (start == 0)
		{
			start = time_in_ms();
		}
	}
	printf("\n");

	/* Report achieved tok/s (pos-1 because the timer starts after first iteration) */
	if (pos > 1 && start != 0)
	{
		long end = time_in_ms();
		long elapsed = end - start;

		if (elapsed > 0)
		{
			fprintf(stderr, "Achieved tokens/s: %f\n", (pos - 1) * 1000.0 / (double)elapsed);
		}
	}

	free(prompt_tokens);
}

/* Prints the Schnecke help */
void error_usage(void)
{
	fprintf(stderr, "Usage:   schnecke <checkpoint> [options]\n");
	fprintf(stderr,
		"Example: schnecke model.bin -n 256 -i \"Once upon a time\"\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
	fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in "
			"[0,1] default 0.9\n");
	fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
	fprintf(stderr, "  -n <int>    number of steps to run for, default "
			"256. 0 = max_seq_len\n");
	fprintf(stderr, "  -i <string> input prompt\n");
	fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
	fprintf(stderr, "  -h          This help\n");
	exit(EXIT_FAILURE);
}

/* Prints the Schnecke banner */
void print_banner(void)
{
	printf("Schnecke LLM Inference Engine v1.1.0\n");
	printf("For AmigaOS and classic systems\n\n");
	printf("Originally based on llama2.c by Andrej Karpathy\n");
	printf("AmigaOS/68k port and modifications by Jani Kuituniemi\n\n");
	printf("Build date: Nov 2026\n\n");
	printf("Run with -h for help.\n\n");
}

int main(int argc, char *argv[])
{
	/* Default parameters */
	char *checkpoint_path = NULL;
	char *tokenizer_path = "tokenizer.bin";
	float temperature = 1.0f;
	float topp = 0.9f; /*  top-p in nucleus sampling. */
	int steps = 256; 
	char *prompt = "";
	unsigned long rng_seed = 0; /* Seed rng with time by default */
	Transformer transformer;
	Tokenizer tokenizer;
	Sampler sampler;

	int i;
	if (argc > 1)
	{
		if (strcmp(argv[1], "-h") == 0)
		{
			error_usage();
		}
	}
	if (argc >= 2)
	{
		checkpoint_path = argv[1];
	}
	else
	{
		error_usage();
	}
	for (i = 2; i < argc; i += 2)
	{
		/* Do some basic validation */
		if (i + 1 >= argc)
		{
			error_usage();
		} /* Must have arg after flag */
		if (argv[i][0] != '-')
		{
			error_usage();
		} /* Must start with dash */
		if (strlen(argv[i]) != 2)
		{
			error_usage();
		} 
		/* Must be -x (one dash, one letter) 
		  read in the args */
		if (argv[i][1] == 't')
		{
			temperature = atof(argv[i + 1]);
		}
		else if (argv[i][1] == 'p')
		{
			topp = atof(argv[i + 1]);
		}
		else if (argv[i][1] == 's')
		{
			rng_seed = atoi(argv[i + 1]);
		}
		else if (argv[i][1] == 'n')
		{
			steps = atoi(argv[i + 1]);
		}
		else if (argv[i][1] == 'i')
		{
			prompt = argv[i + 1];
		}
		else if (argv[i][1] == 'z')
		{
			tokenizer_path = argv[i + 1];
		}
		else
		{
			error_usage();
		}
	}

	/* Print program banner */
	print_banner();

	/* Parameter validation/overrides */
	if (rng_seed <= 0)
		rng_seed = (unsigned int)time(NULL);
	if (temperature < 0.0)
		temperature = 0.0;
	if (topp < 0.0 || 1.0 < topp)
		topp = 0.9;
	if (steps < 0)
		steps = 0;

	/* Build the Transformer via the model .bin file */
	build_transformer(&transformer, checkpoint_path);
	if (steps == 0 || steps > transformer.config.seq_len)
		steps = transformer.config.seq_len; /* Override to ~max length */

	/* Build the Tokenizer via the tokenizer .bin file */
	build_tokenizer(&tokenizer, tokenizer_path,
			transformer.config.vocab_size);

	/*  Build the Sampler */
	build_sampler(&sampler, transformer.config.vocab_size, temperature,
		      topp, rng_seed);

	/* Run! */
	generate(&transformer, &tokenizer, &sampler, prompt, steps);

	/* Memory and file handles cleanup */
	free_sampler(&sampler);
	free_tokenizer(&tokenizer);
	free_transformer(&transformer);
	return 0;
}
