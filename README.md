<div align="center">

# DeepHAM: A global solution method for heterogeneous agent models with aggregate shocks

Jiequn Han, Yucheng Yang, Weinan E

[![arXiv](https://img.shields.io/badge/arXiv-2112.14377-b31b1b.svg)](https://arxiv.org/abs/2112.14377)
[![SSRN](https://img.shields.io/badge/SSRN-3990409-133a6f.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3990409)

</div>


## Dependencies
* Quick installation of conda environment for Python: ``conda env create -f environment.yml``

## Running
### Quick start for the Krusell-Smith model under default configs:
To train a competitive equilibrium solution for the KS model, run
```
train_KS.py
```
To evaluate the Bellman error of the solution for KS model
```
validate_KS.py
```

Sample scripts for solving the KS model in the Slurm system are provided in the folder ``src/slurm_scripts``

### Solve competitive equilibrium solutions for the model in Fernandez-Villaverde et al. (2019)
```
train_JFV.py
```
```
validate_JFV.py
```

## Citation
If you find this work helpful, please consider starring this repo and citing our paper using the following Bibtex.
```bibtex
@article{han2021deepham,
  title={Deepham: A global solution method for heterogeneous agent models with aggregate shocks},
  author={Han, Jiequn and Yang, Yucheng and E, Weinan},
  journal={arXiv preprint arXiv:2112.14377},
  year={2021}
}
```

## Contact
Please contact us at jiequnhan@gmail.com and yucheng.yang@bf.uzh.ch if you have any questions.
