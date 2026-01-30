import glob
import os
import re

scripts = glob.glob(r"c:\Users\Ramss\Desktop\Proyectos\gender_diaparity\src_stat\publication\*.py")

for s in scripts:
    with open(s, 'rb') as f:
        content = f.read()
    
    # Strip non-ASCII
    clean_content = "".join([chr(b) for b in content if b < 128])
    
    with open(s, 'w', encoding='ascii') as f:
        f.write(clean_content)
    print(f"ASCII-fied {s}")
