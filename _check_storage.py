import shutil, string, os
GB = 1024**3
print("{:<8}{:>12}{:>12}{:>12}{:>8}".format("Drive","Total(GB)","Used(GB)","Free(GB)","Free%"))
for d in string.ascii_uppercase:
    root = d + ":\\"
    if os.path.exists(root):
        try:
            t, u, f = shutil.disk_usage(root)
            print("{:<8}{:>12.1f}{:>12.1f}{:>12.1f}{:>7.1f}%".format(root, t/GB, u/GB, f/GB, f/t*100))
        except Exception as e:
            print(root, "err", e)
