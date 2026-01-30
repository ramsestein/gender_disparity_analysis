import glob
import os

scripts = glob.glob(r"c:\Users\Ramss\Desktop\Proyectos\gender_diaparity\src_stat\publication\p*.py")

emojis = {
    "\u2705": "[OK]",
    "\u231b": "[WAIT]",
    "\ud83c\udfa8": "[PLOT]",
    "\u26a0\ufe0f": "[WARN]",
    "\u274c": "[FAIL]",
    "\ud83d\ude80": "[START]",
    "\u2728": "[DONE]",
    "\ud83c\udfc6": "[MASTER]"
}

for s in scripts:
    with open(s, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for e, r in emojis.items():
        new_content = new_content.replace(e, r)
        
    if new_content != content:
        with open(s, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Cleaned {s}")
