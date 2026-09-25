import zipfile, os, time

ZIP = r"C:\Users\Akashat\Downloads\6ab10eb3b23ba_student_resource.zip"
DEST = r"D:\CODE SNAP\ml_challenge"   # data + scratch live here (190 GB free)

os.makedirs(DEST, exist_ok=True)
z = zipfile.ZipFile(ZIP)

def keep(name):
    if "__MACOSX" in name:
        return False
    base = os.path.basename(name)
    if base.startswith("._") or base == ".DS_Store":
        return False
    if name.endswith("/"):
        return False
    return True

targets = [i for i in z.infolist() if keep(i.filename)]
total = sum(i.file_size for i in targets)
print("Files to extract:", len(targets))
print("Total uncompressed: {:,} bytes (~{:.2f} GB)".format(total, total/1024**3))
print("Destination:", DEST)
print("-" * 60)

done = 0
t0 = time.time()
for info in targets:
    # strip leading "student_resource/" so we get D:\ml_challenge\dataset\...
    rel = info.filename
    prefix = "student_resource/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    out_path = os.path.join(DEST, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with z.open(info) as src, open(out_path, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)  # 1 MB streaming, low memory
            if not chunk:
                break
            dst.write(chunk)
    done += info.file_size
    print("  [{:>5.1f}%] {}".format(done / total * 100, rel))

print("-" * 60)
print("DONE in {:.1f}s".format(time.time() - t0))
