# Schnecke_LLM

![Schnecke LLM Logo](https://github.com/jkuituni/Schnecke_LLM/images/Schnecke_LLM.jpg "")

Schnecke is a portable C89-based LLM inference engine for AmigaOS and classic systems, based on Andrej Karpathy’s llama2.c. It was created primarily as an experiment to see whether modern transformer inference could be made to run locally on vintage 68k Amiga hardware.

*"Because sometimes the question is not whether something should be done, but whether it could be done."*

##Features

* Portable single-file C89 codebase
* AmigaOS / Motorola 68k compatible
* Endian-safe checkpoint loading
* FP32 inference support
* TinyStories 15M verified working
* Minimal dependencies
* Based on llama2.c

##Why?

Schnecke began primarily as an experiment into local AI inference on classic hardware. The goal was not speed or practicality, but understanding what modern transformer inference fundamentally requires and how it actually works under the hood. Schnecke has been very valuable as an educational tool and journey. While Schnecke was developed primarily for AmigaOS and Motorola 68k systems, an effort was made to keep the code portable and dependency-light using mostly old-style C89-compatible code. Especially some old UNIX workstations would be an interesting target as they often have quite reasonable amount of CPU power but are similarly to the Amiga constrained by old versions of GCC compiler or even use the old K&R C compiler.

##Screenshot

![Schnecke LLM running on Amiga](https://github.com/jkuituni/Schnecke_LLM/images/Schnecke_Workbench.jpg "")

##Requirements

While exact requirements depend on compile flags, I strongly recommended to have the fastest 68060 accelerator you can get your hands on.

##Hardware

* Motorola 68060 accelerator
* Accelerator local Fast RAM essential
* ~100 MB RAM for TinyStories 15M FP32
* Patience
* Coffee

##Software

* AmigaOS 3.x
* GeekGadgets GCC environment
* GCC 2.95+
* libm

##Building

*`gcc -noixemul -std=c89 -O2 schnecke.c -lm -o schnecke`*

Compiler optimization flags such as -O2 are strongly recommended due to the computational cost of inference on classic hardware.

##Running

To run Schnecke, you need both the actual model as well as the Tokenizer. To get the model, try downloading stories15M model from huggingface;

    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

Other llama2.c-compatible models may also work provided sufficient RAM is available and patience to go with it. Andrej hosts a number of smaller and bigger models on his page.

And for the Tokenizer, use the one from Andrej's repository here;

    https://github.com/softmax1/llama2.c-tinystories

Once those are in place in the same directory where you downloaded Schnecke to, you can run using for example the following command;

    schnecke stories15M.bin -t 0.8 -n 256 -i "Once upon a time"

You should receive something like this;

> Once upon a time there was a little girl named Lily. She loved to play with her toys and her friends. One day, Lily and her friends decided to have a race. They ran and ran until they reached a big tree. They were both very competitive and wanted to win. Lily's friend, Timmy, said, "I bet I can run faster than you!" but Lily replied, "I'm not the fastest, I'm just tall enough to beat you!" They both started to run as fast as they could. As they were running, Lily accidentally bumped into a small boy named Max. Max got upset and started to cry. Lily said, "I'm sorry, Max. I didn't mean to bump into you. Let's be friends and play together." Max stopped crying and smiled. From that day on, they were the best of friends and always played together.

##Parameters:

* -t : Temperature in [0,inf], default 1.0
* -p : p value in top-p (nucleus) sampling in [0,1] default 0.9
* -s : Random seed, default time(NULL)
* -n : Number of steps to run for, default 256. 0 = max_seq_len
* -z : Optional path to custom tokenizer, default $PROGDIR
* -h : Prints Help.

##Benchmarks

| System        | Model           | Precision | tokens/s | Approx. sec/token |
|---------------|-----------------|-----------|----------|-------------------|
| A4000 / 68060 | TinyStories 15M | FP32      | 0.046    | ~21s              |

This project is not intended to be fast. The purpose is experimentation, portability and technical curiosity.

##Known Limitations

* FP32 only (currently)
* Slow inference on classic hardware
* Limited model compatibility
* No GPU acceleration
* Minimalist architecture by design

##Future Ideas

* FP16/Q8/Q4 support
* Additional checkpoint formats
* Optimization experiments
* More portable backends
* Benchmark comparisons

##Credits

Schnecke is based heavily on Andrej Karpathy’s excellent llama2.c project.

Original project:
[https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)

##License

Schnecke is licensed under the MIT License.

See LICENSE for details.
