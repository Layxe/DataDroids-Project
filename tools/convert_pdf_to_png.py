import argparse
import os
from tqdm import tqdm
from pdf2image import convert_from_path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert all pdf files within subfolders to png using pdf2image.")
    parser.add_argument("input", type=str, help="path to folder containing pdf files")
    args = parser.parse_args()
    return args

def main(args):
    
    file_count = sum(len(files) for _, _, files in os.walk(args.input))
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(args.input):
            for file in files:
                # convert file to png if pdf
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    png_path = pdf_path.replace(".pdf", ".png")
                    # convert
                    img = convert_from_path(pdf_path)
                    img[0].save(png_path, 'PNG')
                    

if __name__ == '__main__':
    args = parse_args()
    main(args)