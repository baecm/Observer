import os
import glob
from argparse import ArgumentParser

from utils import DataLoader, Preprocessor, Converter, Resolver
from utils import TempFileManager

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help="Input file or directory containing .rep files")
    parser.add_argument('--output', type=str, help="Output directory for processed files")
    args = parser.parse_args()
    return args


def resolve_input_paths(input_path):
    # TEST: If input_path is None, find and return .rep files from the data folder.
    if input_path is None:
        data_root = os.path.join(os.getcwd(), 'data')
        rep_files = glob.glob(os.path.join(data_root, '*.rep'))
        return rep_files
    
    if os.path.isdir(input_path):
        rep_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.rep')]
    elif os.path.isfile(input_path) and input_path.endswith('.rep'):
        rep_files = [input_path]
    else:
        raise ValueError("Invalid input: Must be a .rep file or a directory containing .rep files")

    if not rep_files:
        raise ValueError("No .rep files found in the specified input directory")

    return rep_files


def main():
    args = parse_arguments()
    
    rep_files = resolve_input_paths(args.input)
    
    for rep_file in rep_files:
        loader = DataLoader(rep_file, args)
        
        temp_manager = TempFileManager(loader.data['temp_dir'])

        with Preprocessor(loader.data, args) as preprocessor:
            preprocessor.run()

        with Converter(loader.data, args) as converter:
            converter.run()
            
        with Resolver(loader.data['temp_dir'], loader.data['resolution_frame'], args.output) as resolver:
            resolver.run()
        
        temp_manager.cleanup()
            


if __name__ == '__main__':
    main()