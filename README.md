<h1> ChatEMG:
Synthetic Data Generation to Control a Robotic Hand Orthosis for
Stroke</h1>
<div style="text-align: center;">

[Jingxi Xu](https://jxu.ai/)$^{* , 1}$, [Runsheng Wang](https://jxu.ai/chatemg/)$^{* 
, 1}$, [Siqi Shang](https://jxu.ai/chatemg/)$^{*
,1}$, [Ava Chen](https://avachen.net/)$^1$, [Lauren Winterbottom](#)$^2$, [To-Liang Hsu](#)$^1$, [Wenxi Chen](#)$^1$, [Khondoker Ahmed](#)$^1$,
[Pedro Leandro La Rotta](#)$^1$, [Xinyue Zhu](#)$^1$, [Dawn M. Nilsen](#)$^2$, [Joel Stein](#)$^2$, [Matei Ciocarlie](https://roam.me.columbia.edu/people/matei-ciocarlie)
$^1$

$^*$ Equal contribution, $^1$ Columbia University, $^2$ Columbia University Irving Medical Center

[Project Page](https://jxu.ai/chatemg/) | [Arxiv](https://arxiv.org/abs/2406.12123) | [Video](https://www.youtube.com/watch?si=wWuCxBVVM1tPidTz&v=ozLbAGEkCug&feature=youtu.be)

<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="docs_code/assets/teaser.gif">   

ChatEMG is an autoregressive generative model that can generate synthetic EMG signals conditioned on prompts (i.e., a
given sequence of EMG signals). ChatEMG enables us to collect only a small dataset from the new condition, session, or
subject and expand it with synthetic samples conditioned on prompts from this new context.

</div>
</div>

This repository contains code for model training and synthetic EMG data generation
for [ChatEMG](https://jxu.ai/chatemg/).

If you find this codebase useful, consider citing:

```bibtex
@article{xu2024chatemg,
  title={ChatEMG: Synthetic Data Generation to Control a Robotic Hand Orthosis for Stroke},
  author={Xu, Jingxi and Wang, Runsheng and Shang, Siqi and Chen, Ava and Winterbottom, Lauren and Hsu, To-Liang and Chen, Wenxi and Ahmed, Khondoker and La Rotta, Pedro Leandro and Zhu, Xinyue and others},
  journal={arXiv preprint arXiv:2406.12123},
  year={2024}
}
```

If you have any questions, please contact [Jingxi](https://jxu.ai) at `jxu [at] cs [dot] columbia [dot] edu`.

**Table of Contents**

- ⚙️ [Setup](docs_code/setup.md)
- 🚶 [Codebase Walkthrough](docs_code/walkthrough.md)
    - 💾 [EMG Data](docs_code/walkthrough.md#emg-data)
    - ✌️ [Two Branches](docs_code/walkthrough.md#two-branches)
    - 🔄 [Channel Rotation](docs_code/walkthrough.md#channel-rotation)
- 🔬 [Reproducing](docs_code/reproduce.md)
    - 🧠 [Training](docs_code/reproduce.md#training)
    - 📊 [Evaluation](docs_code/reproduce.md#evaluation)

# Acknowledgements

This work was supported in part by the National Institutes of Health (R01NS115652, F31HD111301) and the CU Data Science
Institute.

## Code

- [NanoGPT](https://github.com/karpathy/nanoGPT): The codebase is based on the NanoGPT repository.