from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main():
    
    datafolder = '../data/ptbxl/'
    datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

    models = [
        # conf_fastai_xresnet1d101_6lead,
        # conf_fastai_xresnet1d101_3lead,
        # conf_fastai_xresnet1d101_5lead,
        # conf_fastai_xresnet1d101_7lead,
        # conf_fastai_xresnet1d101_1lead,
        # conf_fastai_xresnet1d101_1_2lead,
        conf_fastai_xresnet1d101,
        # conf_fastai_resnet1d_wang,
        # conf_fastai_lstm,
        # conf_fastai_lstm_bidir,
        # conf_fastai_fcn_wang,
        # conf_fastai_inception1d,
        # conf_wavelet_standard_nn,
        ]

    ##########################################
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('custom_exp', 'normabnorm'),
        # ('exp0', 'all'),
        # ('exp1', 'diagnostic'),
        # ('exp1.1', 'subdiagnostic'),
        # ('exp1.1.1', 'superdiagnostic'),
        # ('exp2', 'form'),
        # ('exp3', 'rhythm')
       ]

    for name, task in experiments:
        print(f'Running experiment {name} for task {task}')
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        print(f'Experiment {name} created')
        e.prepare()
        print('Preparation done')
        e.perform()
        print('Performance done')
        # e.evaluate()
        # print('Evaluation done')

    # # generate greate summary table
    # utils.generate_ptbxl_summary_table()

if __name__ == "__main__":
    main()
