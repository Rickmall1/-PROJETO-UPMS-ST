import os
import random
import subprocess

# Lista arquivos da pasta "instancias"
arquivos = [f for f in os.listdir('instancias/Teste') if f.endswith('.txt')]

# Embaralha a lista
random.shuffle(arquivos)

# Itera e executa o script para cada arquivo
for nome in arquivos:
    print(nome)
    i = nome.find(".")
    caminho = os.path.join('instancias/Teste', nome)  # caminho completo

    comando = [
        "python", "grafico.py", caminho, 
        "MNNO;V;p;S", "resultadosSmall.csv", 
        f"Gantt/{nome[:i]}.png", "1", "300"
    ]
    processo = subprocess.Popen(comando)
    print(f"Iniciado processo para {nome} (PID: {processo.pid})")

print("Todos os comandos foram enviados para execução em segundo plano.")

#"python", "grafico.py", [instancia], "MNNO;V;p;S", "resultadosSmall.csv", "Gantt/[instancia].png", "1", "300"
#"python", "variaveis.py", [instancia], "MNNO;V;p;S", "resultadosSmall.csv", "Gantt/[instancia].png", "1", "300"