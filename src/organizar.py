import os
import shutil

root = "data"   # ajuste se sua pasta for diferente

# mapeia nomes para as classes (prefixos)
classes = ["homer", "bart", "lisa", "marge", "maggie"]

def organizar(base):
    base_path = os.path.join(root, base)

    if not os.path.exists(base_path):
        print(f"Pasta não existe: {base_path}")
        return

    # cria pastas das classes dentro de Train ou Valid
    for cls in classes:
        cls_path = os.path.join(base_path, cls)
        os.makedirs(cls_path, exist_ok=True)

    # move arquivos
    for file in os.listdir(base_path):
        if not file.lower().endswith(".bmp"):
            continue

        prefix = file.split('_')[0]  # pega "homer", "bart", etc.

        # arquivo está no formato homer057.bmp ou bart113.bmp etc
        # então prefix = homer057  → precisamos extrair somente o nome do personagem
        for cls in classes:
            if file.lower().startswith(cls):
                src = os.path.join(base_path, file)
                dst = os.path.join(base_path, cls, file)
                print(f"Movendo {src} → {dst}")
                shutil.move(src, dst)
                break

# organizar train e valid
organizar("Train")
organizar("Valid")

print("✔ Organização concluída!")
