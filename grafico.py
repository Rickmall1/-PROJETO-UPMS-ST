from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD, GUROBI
import gurobipy
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
import random

dados={
    #Definição dos conjuntos
    "M":[], #Máquinas
    "N":[], #Jobs
    "N0":[], #Jobs + dummy 0
    #Constantes
    "V":0, #Constante grande
    #Parâmetros
    "p":{}, #p[ij]: Tempo de processamento do job j na máquina i
    "S":{} #S[i,j,k]: Tempo de setup do job j=>k na máquina i
}

def gantt_chart(M,N,N0,X,C,p,nome):
    # Criar a figura
    fig, ax = plt.subplots(figsize=(10, 5))

    # Cores para os jobs
    colors = plt.cm.get_cmap("tab10", len(N))

    # Resultados da resolução (substituir pelos valores obtidos do solver)
    C_values = { (i, j): value(C[i, j]) for i in M for j in N }  # Pegamos o valor numérico

    # Plotar os retângulos no gráfico
    for i in M:
        for j in N:
            for k in N0:
                if j != k:
                    if (i,k,j) in X.keys():
                        if X[i,k,j] > 0.5:
                            #print(f'Job {k} precede {j} na máquina {i}')
                            # Tempo de início do job
                            start_time = C_values[i, j] - p[i, j]
                            ax.barh(y=i, width=p[i, j], left=start_time, height=0.4, color=colors(j-1), edgecolor='black', label=f'Job {j}' if i == 1 else "")

                            # Adicionar labels
                            ax.text(start_time + p[i, j]/2, i, f'J{j}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

    # Configurações do gráfico
    ax.set_yticks(M)
    ax.set_yticklabels([f'Máquina {i}' for i in M])
    ax.set_xlabel("Tempo")
    ax.set_title("Gráfico de Gantt - Sequenciamento de Jobs")

    ax.set_xticks(np.arange(0, max(C_values.values()) + 2, 1))  # Marcas de 1 em 1 no eixo x
    ax.grid(axis='x', linestyle="--", alpha=0.7)  # Ativar grid no eixo x

    # Evitar repetição de labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Mostrar o gráfico
    plt.savefig(nome, dpi=300, bbox_inches='tight')
    plt.show() #FUNCIONANDO CORRETAMENTE

def salvar_resultado(
    arquivo_saida: str,
    instancia: str,
    modelo: str,
    objetivo: float,
    lowerbound,  # Pode ser float ou None
    gap,         # Pode ser float ou None
    runtime: float,
    constraints: int,
    variables: int,
    status: str  
):
    # Verifica se o arquivo já existe para saber se deve escrever o cabeçalho
    novo_arquivo = not os.path.exists(arquivo_saida)

    # Se lowerbound ou gap forem None, vamos definir uma string padrão '---'
    lowerbound_str = f"{lowerbound:.4f}" if lowerbound is not None else "---"
    gap_str = f"{gap:.2f}%" if gap is not None else "---"

    index = instancia.find("\\")
    
    with open(arquivo_saida, "a") as f:
        # Escreve o cabeçalho se o arquivo for novo
        if novo_arquivo:
            f.write("INSTANCE\tMODEL\tOBJECTIVE\tLOWERBOUND\tGAP\tRUNTIME\tCONSTRAINTS\tVARIABLES\tSTATUS\n")

        # Formata a linha com os valores, separando por tabulação
        linha = (f"{instancia[index+1:]}\t"
                 f"{modelo}\t"
                 f"{objetivo:.4f}\t"
                 f"{lowerbound_str}\t"
                 f"{gap_str}\t"
                 f"{runtime:.4f}\t"
                 f"{constraints}\t"
                 f"{variables}\t"
                 f"{status}\n")
        f.write(linha)

def modelSolving(M, N, N0, p, S, args, time_limit=None):

  #Constante grande:
  V = 1000

  model = LpProblem(name="UPMSPST", sense=LpMinimize)

  #Variáveis de decisão:
  X = LpVariable.dicts("X", product(M, N0, N), cat='Binary')
  C = LpVariable.dicts("C", product(M, N0), lowBound=0, cat='Continuous')

  Cmax = LpVariable("Cmax", lowBound=0, cat='Continuous')

  #Função objetivo (1):
  model += Cmax

  #Restrições (2):
  for k in N:
    model += lpSum(X[i, j, k] for i in M for j in N0 if j != k) == 1

  #Restrições (3):
  for j in N:
    model += lpSum(X[i, j, k] for i in M for k in N if j != k) <= 1

  #Restrições (4):
  for i in M:
    model += lpSum(X[i, 0, k] for k in N) <= 1

  #Restrições (5):
  for j in N:
    for k in N:
      if j != k:
        for i in M:
          model += lpSum(X[i, h, j] for h in N0 if h != k and h != j) >= X[i, j, k]

  #Restrições (6):
  for j in N0:
    for k in N:
      if j != k:
        for i in M:
          model += C[i,k] + V*(1 - X[i, j, k]) >= C[i,j] + S.get((i, j, k), 0) + p.get((i, k), 0)

  #Restrições (7):
  for i in M:
    model += C[i,0] == 0

  #Restrições (8):
  for i in M:
    for j in N:
      model += C[i, j] >= 0

  #Restrições (9):
  for j in N:
    for i in M:
      model += Cmax >= C[i,j]


  if time_limit is not None:
    modelo=model.solve(GUROBI(msg=True, logPath="gurobi.log", logFile=args.nome_instancia + ".log", timeLimit=time_limit))
  else:
    modelo=model.solve(GUROBI(msg=True, logPath="gurobi.log", logFile=f"{args.nome_instancia}.log"))

  #Dicionario com os valores de X e C:
  X_val = {k: v.varValue for k, v in X.items()}
  C_val = {k: v.varValue for k, v in C.items()}
  #Valor otimo do makespan
  Cmax_val = Cmax.varValue

  if isinstance(model.solverModel, gurobipy.Model):
    lower_bound = model.solverModel.ObjBound
    mip_gap = model.solverModel.MIPGap
    status = model.solverModel.Status
  else:
    lower_bound = None
    mip_gap = None
    status = None
  return X_val, C_val, Cmax_val, modelo, value(model.objective), lower_bound, mip_gap, model.solutionTime, len(model.constraints), len(model.variables()), status

def geraInstancia(seed, M, N, lowerP, upperP, lowerS, upperS):
    global dados
    random.seed(seed)

    #Criando uma lista com os valores passados pra M, N e N0:
    maquinas = list(range(1, M+1))
    jobs = list(range(1, N+1))
    N0 = [0] + jobs #adicona o dummy na lista

    #Gerando tempo de processamento aleatório:
    p = {}
    for i in maquinas:
        for j in jobs:
            if j == 0:
                p[(i, j)] = 0
            else:
                p[(i, j)] = random.randint(lowerP, upperP)

    #Gerando tempo de setup aleatório:
    S = {}
    for i in maquinas:
        for j in N0:
            for k in jobs:
                if j != k:
                    S[(i, j, k)] = random.randint(lowerS, upperS)

    dados["M"]=maquinas
    dados["N"]=jobs
    dados["N0"]=N0
    dados["V"]=0
    dados["p"]=p
    dados["S"]=S
    return

#Identifica o tipo de arquivo
def recebe_tipo_arquivo(formato):
    if formato=="json" or formato=="JSON":
        return 1
    elif formato=="dat" or formato=="DAT":
        return 3
    elif formato=="rand" or formato=="RAND":
        return 4
    else:
        return 2
    
#Gerencia a leitura de arquivos
def le_arquivo(args):
    if not(os.path.exists(args.nome_instancia)) and (args.formato!="rand" or args.formato!="RAND"):
        print(f"O arquivo <{args.nome_instancia}> não existe!")
        exit()
    tipo=recebe_tipo_arquivo(args.formato)
    if tipo==1:
        le_arquivo_json(args.nome_instancia)
    elif tipo==2:
        le_formato_variavel(args.nome_instancia, args.formato)
    elif tipo==3:
        le_arquivo_dat(args.nome_instancia)
    elif tipo==4:
        seed=int(input("Digite a seed do gerador aleatório: "))
        M=int(input("Digite a quantidade de máquinas paralelas: "))
        N=int(input("Digite a quantidade de jobs: "))
        lowerP=int(input("Digite o lower P: "))
        upperP=int(input("Digite o UpperP: "))
        lowerS=int(input("Digite o lower S: "))
        upperS=int(input("Digite o UpperS: "))
        geraInstancia(seed,M,N, lowerP, upperP, lowerS, upperS)
    else:
        print("Formato de instancia não suportada!\nTente <dat>,<json> ou <txt>")
        exit()
    return

#Lê a instância do JSON e o armazena em dados
def le_arquivo_json(diretorio):
    global dados
    formato_referencia = {"M", "N", "N0", "V", "p", "S"}

    with open(diretorio, "r") as arquivo:
        dados_carregados = json.load(arquivo)

    if not formato_referencia.issubset(dados_carregados.keys()):
        print(f"A instancia <{diretorio}> não está formatada!")
        exit()

    # Converter chaves de p e S de string para tupla
    dados_carregados["p"] = {
        tuple(map(int, chave.split(","))): valor
        for chave, valor in dados_carregados["p"].items()
    }

    dados_carregados["S"] = {
        tuple(map(int, chave.split(","))): valor
        for chave, valor in dados_carregados["S"].items()
    }

    dados = dados_carregados
    return dados

#Lê de acordo com string variável

def le_formato_variavel(diretorio, formato):
    global dados

    ordem = formato.split(";")

    with open(diretorio, "r") as f:
        linhas = f.read().splitlines()

    linha_atual = 0

    for campo in ordem:
        if campo == "MNNO":
            # Aqui invertemos M e N, pois o formato do arquivo é: N M N0
            N, M, N0 = map(int, linhas[linha_atual].split())
            dados["M"] = list(range(1, M + 1))  # máquinas: 1..M
            dados["N"] = list(range(1, N + 1))  # jobs: 1..N
            dados["N0"] = [0] + dados["N"]      # inclui dummy job 0
            linha_atual += 1

        elif campo == "V":
            V = int(linhas[linha_atual])
            dados["V"] = V
            linha_atual += 1

        elif campo == "p":
            # Leitura dos tempos de processamento da tarefa j na máquina i
            N_ids = dados["N"]
            M_ids = dados["M"]
            p = {}
            for j in N_ids:
                partes = list(map(int, linhas[linha_atual].split()))
                for i in range(0, len(partes), 2):
                    maquina = partes[i] + 1  # índice 1-based
                    tempo = partes[i + 1]
                    p[(maquina, j)] = tempo
                linha_atual += 1
            dados["p"] = p

        elif campo == "S":
            if linhas[linha_atual].strip() != "SSD":
                raise ValueError("Esperado marcador 'SSD' não encontrado")
            linha_atual += 1

            S = {}
            M_ids = dados["M"]
            N_ids = dados["N"]

            for m_index in range(len(M_ids)):
                maquina = M_ids[m_index]

                marcador_maquina = linhas[linha_atual].strip()
                if not marcador_maquina.startswith("M"):
                    raise ValueError(f"Esperado marcador de máquina, obtido '{marcador_maquina}'")
                linha_atual += 1

                for i_index, j in enumerate(N_ids):  # para cada linha da matriz (job j)
                    valores = list(map(int, linhas[linha_atual].split()))
                    for k_index, k in enumerate(N_ids):  # para cada coluna da matriz (job k)
                        S[(maquina, j, k)] = valores[k_index]
                    linha_atual += 1
            dados["S"] = S


#Lê a instância do dat e o armazena em dados
def le_arquivo_dat(diretorio):
    dados = {
        "V": None,
        "p": {},
        "S": {}
    }

    with open(diretorio, "r") as f:
        linhas = f.read().splitlines()

    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()

        # Ignorar linhas vazias
        if not linha:
            i += 1
            continue

        # param V
        if linha.startswith("param V"):
            dados["V"] = int(linha.split(":=")[1].strip(" ;"))

        # set M;
        elif linha.startswith("set"):
            partes = linha.split(":=")
            nome_set = partes[0].split()[1].strip()
            elementos = partes[1].strip(" ;").split()
            dados[nome_set] = list(map(int, elementos))

        # param p :=
        elif linha.startswith("param p"):
            i += 1
            while not linhas[i].strip().endswith(";"):
                valores = linhas[i].strip().strip(",").split(",")
                for item in valores:
                    partes = item.strip().split()
                    if len(partes) == 3:
                        i_m, j, val = map(int, partes)
                        dados["p"][(i_m, j)] = val
                i += 1
            # Última linha com ;
            valores = linhas[i].strip().strip(";").split(",")
            for item in valores:
                partes = item.strip().split()
                if len(partes) == 3:
                    i_m, j, val = map(int, partes)
                    dados["p"][(i_m, j)] = val

        # param S :=
        elif linha.startswith("param S"):
            i += 1
            while not linhas[i].strip().endswith(";"):
                valores = linhas[i].strip().strip(",").split(",")
                for item in valores:
                    partes = item.strip().split()
                    if len(partes) == 4:
                        i_m, j, k, val = map(int, partes)
                        dados["S"][(i_m, j, k)] = val
                i += 1
            # Última linha com ;
            valores = linhas[i].strip().strip(";").split(",")
            for item in valores:
                partes = item.strip().split()
                if len(partes) == 4:
                    i_m, j, k, val = map(int, partes)
                    dados["S"][(i_m, j, k)] = val

        i += 1

    return dados


def calcula_gap(tempo_a, tempo_n):
    gap=(tempo_n-tempo_a)/tempo_a
    return gap

if __name__ == "__main__":
     #Recebe argumentos do terminal e os armazena em args
    parser=argparse.ArgumentParser(description="Solver UPMS-ST")
    parser.add_argument("nome_instancia", help="Nome da instância")
    parser.add_argument("formato", help="Formato da instância (ex: txt, json)")
    parser.add_argument("arq_saida", help="Arquivo de saída dos resultados")
    parser.add_argument("arq_gantt", help="Arquivo de saída do gráfico Gantt")
    parser.add_argument("tempo_instancia_a", help="Tempo de saida da instância anterior para calcular o GAP")
    parser.add_argument("time_limit", type=int, help="Limite de tempo em segundos")
    args = parser.parse_args()
    le_arquivo(args) #Chama função de leitura
    #print(dados["S"])
    X, C, Cmax, model, objetivo, lowerbound, gap, runtime, constraints, variable, status=modelSolving(dados["M"], dados["N"], dados["N0"],  dados["p"], dados["S"], args, args.time_limit)
    
    if status == 1:
        status = "LOADED"
    elif status == 2:
        status = "OPTIMAL"
    elif status == 3:
        status = "INFEASIBLE"
    elif status == 4:
        status = "INF_OR_UNBD"
    elif status == 5:
        status = "UNBOUNDED"
    elif status == 6:
        status = "CUTOFF"
    elif status == 7:
        status = "ITERATION_LIMIT"
    elif status == 8:
        status = "NODE_LIMIT"
    elif status == 9:
        status = "TIME_LIMIT"
    elif status == 10:
        status = "SOLUTION_LIMIT"
    elif status == 11:
        status = "INTERRUPTED"
    elif status == 12:
        status = "NUMERIC"
    elif status == 13:
        status = "SUBOPTIMAL"
    elif status == 14:
        status = "INPROGRESS"
    elif status == 15:
        status = "USER_OBJ_LIMIT"
    elif status == 16:
        status = "WORK_LIMIT"
    elif status == 17:
        status = "MEM_LIMIT"
    
    salvar_resultado(args.arq_saida, args.nome_instancia, "Gurobi", objetivo, lowerbound, gap, runtime, constraints, variable, status)
    gantt_chart(dados["M"], dados["N"], dados["N0"], X, C, dados["p"], args.arq_gantt)
    #print(f"Cmax: {Cmax}")
    #
    #print("\n Sequencia de tarefas: ")
    #for (i, j, k), val in X.items():
    #    if val == 1:
    #        print(f"X[maquina: {i}, job: {k}] = apos {j}")

    #print("\nTempos de início (C): ")
    #for (i, j), val in sorted(C.items()):
    #    print(f"C[maquina: {i}, job: {j}] = inicio em {val}")