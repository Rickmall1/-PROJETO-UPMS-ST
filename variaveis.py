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

def gantt_chart(M, N, N0, X, C, p, nome):
    import matplotlib.pyplot as plt
    import numpy as np
    from pulp import value

    # Criar a figura
    fig, ax = plt.subplots(figsize=(12, 6))

    # Cores para os jobs
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(N))
    job_colors = {j: cmap(j - 1) for j in N}

    # Resultados da resolução
    C_values = {(i, j): value(C[i, j]) for i in M for j in N}
    X_values = {(i, j, k): value(var) for (i, j, k), var in X.items()}

    for (i, k, j), val in X:
        if val is not None and val > 0.5:
            start_time = C_values[(i, j)] - p[(i, j)]
            ax.barh(y=i, width=p[(i, j)], left=start_time, height=0.4,
                    color=job_colors[j], edgecolor='black')

            # Adicionar labels dentro dos blocos
            ax.text(start_time + p[(i, j)] / 2, i, f'J{j}', ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold')

    # Configurações do gráfico
    ax.set_yticks(M)
    ax.set_yticklabels([f'Máquina {i}' for i in M])
    ax.set_xlabel("Tempo")
    ax.set_title("Gráfico de Gantt - Sequenciamento de Jobs")
    ax.set_xticks(np.arange(0, max(C_values.values()) + 2, 1))
    ax.grid(axis='x', linestyle="--", alpha=0.7)

    # Legenda
    handles = [plt.Rectangle((0, 0), 1, 1, color=job_colors[j]) for j in N]
    labels = [f'Job {j}' for j in N]
    ax.legend(handles, labels, title="Jobs", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    #plt.savefig(nome, dpi=300, bbox_inches='tight')
    #plt.show() ESTÁ COM PROBLEMA


def salvar_resultado(
    arquivo_saida: str,
    instancia: str,
    jobs: int,
    maquinas: int,
    model: str,
    status: str,  
    objective: float,
    lowerbound,  # Pode ser float ou None
    gap,         # Pode ser float ou None
    runtime: float,
    constraints: int,
    variables: int,
    binarias: int,
    inteiras: int,
    continuas: int
):
    import os

    # Verifica se o arquivo já existe para saber se deve escrever o cabeçalho
    novo_arquivo = not os.path.exists(arquivo_saida)

    # Se lowerbound ou gap forem None, vamos definir uma string padrão '---'
    lowerbound_str = f"{lowerbound:.4f}" if lowerbound is not None else "---"
    gap_str = f"{gap:.2f}%" if gap is not None else "---"

    with open(arquivo_saida, "a") as f:
        # Escreve o cabeçalho se o arquivo for novo
        if novo_arquivo:
            f.write("INSTANCE\tJOBS\tMACHINES\tMODEL\tSTATUS\tOBJECTIVE\tLOWERBOUND\tGAP\tRUNTIME\tCONSTRAINTS\tVARIABLES\tBINARY\tINTEGER\tCONTINUOUS\n")

        # Formata a linha com os valores, separando por tabulação
        ind = instancia.find("\\") + 1
        linha = (f"{instancia[ind:]}\t"
                 f"{jobs}\t"   
                 f"{maquinas}\t"   
                 f"{model}\t"
                 f"{status}\t"
                 f"{objective:.4f}\t"
                 f"{lowerbound_str}\t"
                 f"{gap_str}\t"
                 f"{runtime:.4f}\t"
                 f"{constraints}\t"
                 f"{variables}\t"
                 f"{binarias}\t"
                 f"{inteiras}\t"
                 f"{continuas}\n")
        f.write(linha)

def modelSolving(M, N, N0, p, S, log, time_limit=None):

    # Constante grande:
    V = 1000

    model = LpProblem(name="UPMSPST", sense=LpMinimize)

    # Variáveis de decisão:
    X = LpVariable.dicts("X", product(M, N0, N), cat='Binary')
    C = LpVariable.dicts("C", product(M, N0), lowBound=0, cat='Continuous')
    Cmax = LpVariable("Cmax", lowBound=0, cat='Continuous')

    # Função objetivo (1):
    model += Cmax

    # Restrições (2):
    for k in N:
        model += lpSum(X[i, j, k] for i in M for j in N0 if j != k) == 1

    # Restrições (3):
    for j in N:
        model += lpSum(X[i, j, k] for i in M for k in N if j != k) <= 1

    # Restrições (4):
    for i in M:
        model += lpSum(X[i, 0, k] for k in N) <= 1

    # Restrições (5):
    for j in N:
        for k in N:
            if j != k:
                for i in M:
                    model += lpSum(X[i, h, j] for h in N0 if h != k and h != j) >= X[i, j, k]

    # Restrições (6):
    for j in N0:
        for k in N:
            if j != k:
                for i in M:
                    model += C[i, k] + V * (1 - X[i, j, k]) >= C[i, j] + S.get((i, j, k), 0) + p.get((i, k), 0)

    # Restrições (7):
    for i in M:
        model += C[i, 0] == 0

    # Restrições (8):
    for i in M:
        for j in N:
            model += C[i, j] >= 0

    # Restrições (9):
    for j in N:
        for i in M:
            model += Cmax >= C[i, j]

    # Resolver o modelo
    if time_limit is not None:
        modelo = model.solve(GUROBI(msg=True, logPath=str(log).replace(".txt", ""), timeLimit=time_limit))
    else:
        modelo = model.solve(GUROBI(msg=True, logPath=str(log).replace(".txt", "")))

    # Extrair valores das variáveis, separando por tipo:
    bin_vars = {}
    int_vars = {}
    cont_vars = {}

    for var in model.variables():
        # 'var' é um objeto LpVariable
        if var.cat == 'Binary':
            bin_vars[var.name] = var.varValue
        elif var.cat == 'Integer':
            int_vars[var.name] = var.varValue
        elif var.cat == 'Continuous':
            cont_vars[var.name] = var.varValue

    # Também extrair o valor do Cmax (que é contínuo, mas já está no dicionário cont_vars)
    Cmax_val = Cmax.varValue

    if isinstance(model.solverModel, gurobipy.Model):
        lower_bound = model.solverModel.ObjBound
        mip_gap = model.solverModel.MIPGap
        status = model.solverModel.Status
    else:
        lower_bound = None
        mip_gap = None
        status = None

    return bin_vars, int_vars, cont_vars, modelo, value(model.objective), lower_bound, mip_gap, model.solutionTime, len(model.constraints), len(model.variables()), status

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
    # Recebe argumentos do terminal e os armazena em args
    parser = argparse.ArgumentParser(description="Solver UPMS-ST")
    parser.add_argument("nome_instancia", help="Nome da instância")
    parser.add_argument("formato", help="Formato da instância (ex: txt, json)")
    parser.add_argument("arq_saida", help="Arquivo de saída dos resultados")
    parser.add_argument("arq_gantt", help="Arquivo de saída do gráfico Gantt")
    parser.add_argument("tempo_instancia_a", help="Tempo de saída da instância anterior para calcular o GAP")
    parser.add_argument("time_limit", type=int, help="Limite de tempo em segundos")
    args = parser.parse_args()

    le_arquivo(args)  # Chama função de leitura
    log = "../" + args.nome_instancia + ".log"

    # Chamada atualizada da função modelSolving
    X, X_int, X_cont, model, objetivo, lowerbound, gap, runtime, constraints, variable, status = modelSolving(
        dados["M"], dados["N"], dados["N0"], dados["p"], dados["S"], log, args.time_limit
    )

    # Processar status numérico para string
    status_dict = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }
    status = status_dict.get(status, str(status))  # Garante que não quebre caso status seja desconhecido

    salvar_resultado(
    args.arq_saida,
    args.nome_instancia,
    len(dados["N"]),
    len(dados["M"]),
    "Gurobi",
    status,
    objetivo,
    lowerbound,
    gap,
    runtime,
    constraints,
    variable,
    binarias=len(X),
    inteiras=len(X_int),
    continuas=len(X_cont)
)

    # Montar dicionário C a partir de X_cont
    C = {}
    for var_name, val in X_cont.items():
        if var_name.startswith("C_"):
            # Nome no formato C_(i,_j), precisamos extrair os índices
            indices = var_name[2:].strip("()").split(",_")
            if len(indices) == 2:
                try:
                    i = int(indices[0])
                    j = int(indices[1])
                    C[(i, j)] = val
                except ValueError:
                    pass

    # Gerar gráfico Gantt
    gantt_chart(dados["M"], dados["N"], dados["N0"], X, C, dados["p"], args.arq_gantt)
