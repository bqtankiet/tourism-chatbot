import os

def load_documents(
    folder_path,
    ignore_prefix="00_",
    file_extension=".md"
):
    documents = {}

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not exist: {folder_path}")

    files = os.listdir(folder_path)

    for f_name in files:
        # skip file không đúng định dạng
        if file_extension and not f_name.endswith(file_extension):
            continue

        # skip file không cần thiết
        if ignore_prefix and f_name.startswith(ignore_prefix):
            continue

        path = os.path.join(folder_path, f_name)

        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            documents[f_name] = f.read()

    return documents