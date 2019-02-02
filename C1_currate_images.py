import pixelhouse as ph
import glob, random, os

F_SAMPLES = glob.glob("samples/images/*")
random.shuffle(F_SAMPLES)

f_tags = "analysis/tagged_images.csv"

if os.path.exists(f_tags):
    df = pd.read_csv(f_tags)

f_img = None
load_new = True

save_dest = 'samples/curated/'


save_dest_weird = os.path.join(save_dest, 'weird')
save_dest_hq = os.path.join(save_dest, 'high_quality')
save_dest_boring = os.path.join(save_dest, 'boring')

os.makedirs(save_dest_boring, exist_ok=True)
os.makedirs(save_dest_hq, exist_ok=True)
os.makedirs(save_dest_weird, exist_ok=True)

known = set()
known.update([os.path.basename(f) for f in
              glob.glob(os.path.join(save_dest_weird, '*'))])
known.update([os.path.basename(f) for f in
              glob.glob(os.path.join(save_dest_hq, '*'))])

known.update([os.path.basename(f) for f in
              glob.glob(os.path.join(save_dest_boring, '*'))])


print("Use [0] to mark as boring")
print("Use [1] to mark as high quality")
print("Use [2] to mark as weird")
print("Use [space] to skip")
print("Use [esc] to exit")
print("Use [backspace] to undo")


while F_SAMPLES:

    if load_new:
        f_img = F_SAMPLES.pop(0)
    
    base = os.path.basename(f_img)
    if base in known:
        continue

    load_new = True
    canvas = ph.load(f_img)
    key = canvas.show()

    # Escape or q quits
    if key in [27, 113]:
        print("Quitting")
        break

    # Space, down or right arrow skips
    elif key in [32]:
        continue

    # Backspace, need to undo
    elif key in [8]:
        try:
            print(f"Removing curation label {dst}")        
            os.remove(dst)
            f_img = os.path.join("samples/images/", os.path.basename(dst))
            print(f_img)
        except Exception as EX:
            print(f"Can't remove label: {EX}")
        load_new = False
        
        

    # Mark as weird!
    elif chr(key) == "2":
        print(f"Saving image {f_img} as weird!")
        dst = os.path.abspath(os.path.join(save_dest_weird, base))
        src = os.path.abspath(f_img)
        os.system(f'ln -s {src} {dst}')

    # Mark as high quality!
    elif chr(key) == "1":
        print(f"Saving image {f_img} as high quality!")
        dst = os.path.abspath(os.path.join(save_dest_hq, base))
        src = os.path.abspath(f_img)
        os.system(f'ln -s {src} {dst}')
    
    # Skip the image
    elif chr(key) == "0":
        print(f"Saving image {f_img} as boring...")
        dst = os.path.abspath(os.path.join(save_dest_boring, base))
        src = os.path.abspath(f_img)
        os.system(f'ln -s {src} {dst}')
    else:
        print(f"Key {key}, {chr(key)} not known")
        load_new = False
        
