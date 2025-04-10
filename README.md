```shell
conda init
conda config --set auto_activate_base false
conda create -n srp_test -m python=3.10
codna activate srp_test
pip install torch torchvision torchaudio optuna pandas scikit-learn matplotlib
```