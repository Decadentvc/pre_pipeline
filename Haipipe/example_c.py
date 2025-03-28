from haipipe.HAIPipe import *
import argparse

support_model = ['RandomForestClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']

def quick_start(notebook_path, dataset_path, label_index, model, hai_program_save_path='hai_program.py'):
    """
    This function is a quick start for the HAIPipe.

    Parameters:
    -----------
    notebook_path: str
        The path to the HI-program notebook file.
    dataset_path: str
        The path to the dataset file.
    label_index: int
        The index of the label column in the dataset file.
    model: str
        The model name and now it only supports those in "support_model".
    hai_program_save_path: str
        The path to save the generated HAI-program (固定为 `hai_program.py`).
    """
    
    hai_pipe = HAIPipe(notebook_path, dataset_path, label_index, model)
    hai_pipe.evaluate_hi()
    hai_pipe.generate_aipipe()
    hai_pipe.combine()
    hai_pipe.select_best_hai_by_al()
    hai_pipe.output(hai_program_save_path, save_fig=True)  # 保存路径固定

if __name__ == "__main__":
    # 默认参数
    DEFAULT_NOTEBOOK_PATH = 'data/notebook/datascientist25_gender-recognition-by-voice-using-machine-learning.ipynb'
    DEFAULT_DATASET_PATH = 'data/dataset/primaryobjects_voicegender/voice.csv'
    DEFAULT_LABEL_INDEX = 20
    DEFAULT_MODEL = support_model[2]  # LogisticRegression

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='HAIPipe Quick Start')
    parser.add_argument('--notebook_path', type=str, default=DEFAULT_NOTEBOOK_PATH,
                       help=f'Path to the Jupyter notebook file (default: {DEFAULT_NOTEBOOK_PATH})')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH,
                       help=f'Path to the dataset file (default: {DEFAULT_DATASET_PATH})')
    parser.add_argument('--label_index', type=int, default=DEFAULT_LABEL_INDEX,
                       help=f'Index of the label column (default: {DEFAULT_LABEL_INDEX})')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=support_model,
                       help=f'Model name (default: {DEFAULT_MODEL}), options: {support_model}')

    # 解析参数
    args = parser.parse_args()

    # 调用函数（保存路径固定为默认值，不通过参数传递）
    quick_start(
        args.notebook_path,
        args.dataset_path,
        args.label_index,
        support_model[2],
        hai_program_save_path='hai_program.py'  # 直接固定为默认值
    )