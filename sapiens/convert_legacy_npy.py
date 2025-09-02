#!/usr/bin/env python3
"""
Legacy .npy file converter
Converts old numpy files with incompatible dtypes to modern format
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import tempfile
import shutil

def convert_legacy_npy(input_file, output_file=None):
    """
    Convert a legacy .npy file to modern format
    """
    if output_file is None:
        output_file = input_file + ".converted"

    try:
        # Try to load with numpy 1.21.6 in a subprocess approach
        # or use a direct conversion approach

        # Method 1: Try to read the raw data and reconstruct
        with open(input_file, 'rb') as f:
            # Skip numpy header (usually around 128 bytes, but variable)
            magic = f.read(6)
            if magic != b'\x93NUMPY':
                print(f"Not a valid numpy file: {input_file}")
                return False

            # Read version
            version = f.read(2)

            # Read header length (little endian short)
            header_len_bytes = f.read(2)
            header_len = int.from_bytes(header_len_bytes, 'little')

            # Read header
            header = f.read(header_len).decode('ascii')

            # Try to extract shape and dtype info from header
            try:
                # Parse the header to get basic info
                import ast
                import re

                # Extract the dictionary from the header
                header_dict_str = header.strip().rstrip('\n').rstrip(' ')
                if header_dict_str.endswith(','):
                    header_dict_str = header_dict_str[:-1]

                # Use a safer evaluation
                header_dict = eval(header_dict_str, {"__builtins__": {}},
                                 {"False": False, "True": True, "None": None})

                shape = header_dict['shape']
                fortran_order = header_dict['fortran_order']

                # Read the raw data
                data_bytes = f.read()

                # Determine data type - assume float32 for most depth data
                if 'float' in str(header_dict.get('descr', '')):
                    dtype = np.float32
                elif 'int' in str(header_dict.get('descr', '')):
                    dtype = np.int32
                else:
                    dtype = np.float32  # default assumption for depth data

                # Calculate expected size
                expected_size = np.prod(shape) * np.dtype(dtype).itemsize

                if len(data_bytes) >= expected_size:
                    # Reconstruct array from raw bytes
                    data = np.frombuffer(data_bytes[:expected_size], dtype=dtype)
                    if shape:
                        data = data.reshape(shape)
                        if fortran_order:
                            data = np.asfortranarray(data)

                    # Save in new format
                    np.save(output_file, data.astype(np.float32))
                    return True

            except Exception as e:
                print(f"Could not parse header for {input_file}: {e}")
                return False

    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False

    return False

def convert_directory(input_dir, backup_originals=True):
    """
    Convert all .npy files in a directory
    """
    npy_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))

    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files to convert")

    success_count = 0
    for npy_file in tqdm(npy_files, desc="Converting files"):
        temp_file = npy_file + ".temp_converted"

        if convert_legacy_npy(npy_file, temp_file):
            if backup_originals:
                backup_file = npy_file + ".backup"
                shutil.move(npy_file, backup_file)
            else:
                os.remove(npy_file)

            shutil.move(temp_file, npy_file)
            success_count += 1
        else:
            # Clean up temp file if conversion failed
            if os.path.exists(temp_file):
                os.remove(temp_file)

    print(f"Successfully converted {success_count}/{len(npy_files)} files")
    if backup_originals:
        print("Original files backed up with .backup extension")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_legacy_npy.py <directory_path> [--no-backup]")
        sys.exit(1)

    input_dir = sys.argv[1]
    backup = "--no-backup" not in sys.argv

    if not os.path.isdir(input_dir):
        print(f"Directory not found: {input_dir}")
        sys.exit(1)

    convert_directory(input_dir, backup_originals=backup)

