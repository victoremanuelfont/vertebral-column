% Trabalho RBF - Classificacao Coluna
% Dataset: column_3C.dat

clear; clc;
fprintf('Iniciando ...\n');

   % --- 1. Carregando dados ---
   % O textread separa os numeros das letras (ultima coluna)
arquivo = 'column_3C.dat';
if ~exist(arquivo, 'file')
 error('Arquivo nao encontrado');
end

[col1,col2,col3,col4,col5,col6, coluna_texto] = textread(arquivo, '%f %f %f %f %f %f %s');

% Monta matriz X (junta as colunas numéricas)
dados_entrada = [col1, col2, col3, col4, col5, col6];
coluna_texto = strtrim(coluna_texto);

% Convertendo classes (Texto -> Numero)
% Isso eh necessario pq a rede so faz conta com numero
labels_numericos = zeros(length(coluna_texto),1);
  labels_numericos(strcmp(coluna_texto,'DH')) = 1; % Disk Hernia
  labels_numericos(strcmp(coluna_texto,'SL')) = 2; % Spondilolysthesis
labels_numericos(strcmp(coluna_texto,'NO')) = 3;   % Normal

if any(labels_numericos==0)
  error('Erro na conversao das classes');
end

 % One-Hot Encoding
 % Transforma o numero da classe em vetor [0 1 0]
num_amostras = size(dados_entrada, 1);
 num_classes = 3;
Y_alvo = zeros(num_amostras, num_classes);

for i=1:num_amostras
Y_alvo(i, labels_numericos(i)) = 1;
end

% --- ConfiguraçoÞes da Rede ---
num_execucoes=10;
porcentagem_treino = 0.7;
   num_neuronios = 15;   % Quantos centros vamos usar na camada oculta
sigma = 40;              % Largura da curva gaussiana (ajuste fino)
historico_acuracias = zeros(num_execucoes, 1);

fprintf('Rodando %d execucoes...\n', num_execucoes);

% Loop Principal (executa 10 vezes para tirar a media)
for i = 1:num_execucoes

 % Embaralhar os dados para garantir aleatoriedade na divisao
 indices_embaralhados = randperm(num_amostras);
 X_misturado = dados_entrada(indices_embaralhados, :);
    Y_misturado = Y_alvo(indices_embaralhados, :);

    % Separar Treino (70%) e Teste (30%)
    qtde_treino = round(num_amostras * porcentagem_treino);

  X_treino = X_misturado(1:qtde_treino,:);
  Y_treino = Y_misturado(1:qtde_treino,:);
     X_teste = X_misturado(qtde_treino+1:end,:);
     Y_teste = Y_misturado(qtde_treino+1:end,:);

    % --- RBF Treino ---
    % 1. Escolher Centros: Pegamos amostras aleatorias do próprio treino
    indices_centros = randperm(size(X_treino,1), num_neuronios);
 Centros = X_treino(indices_centros, :);

    % 2. Calcular Matriz H (Camada Oculta)
    % Mede a distancia de cada ponto ate os centros usando funcao Gaussiana
    H = zeros(size(X_treino,1), num_neuronios);
    for k=1:num_neuronios
       diferenca = X_treino - Centros(k,:);
     distancia_quadrada = sum(diferenca.^2, 2);
       H(:,k) = exp(-distancia_quadrada / (2*sigma^2));
    end
   H = [ones(size(X_treino,1), 1), H]; % Adiciona Bias (coluna de 1s)

    % 3. Calcular Pesos W
    % Usa Pseudo-inversa para achar a solucao otima direta
    W = pinv(H) * Y_treino;

   % --- Teste (Validacao) ---
   % Fazemos a mesma conta da matriz H, mas agora com dados de Teste
    H_teste = zeros(size(X_teste,1), num_neuronios);
    for k=1:num_neuronios
        diferenca = X_teste - Centros(k,:);
      distancia_quadrada = sum(diferenca.^2, 2);
        H_teste(:,k) = exp(-distancia_quadrada / (2*sigma^2));
    end
    H_teste = [ones(size(X_teste,1), 1), H_teste];

    % Saida da rede (Previsao)
  Y_estimado = H_teste * W;

    % Pega o indice da maior saida para saber qual a classe (1, 2 ou 3)
    [~, classe_predita] = max(Y_estimado, [], 2);
    [~, classe_real] = max(Y_teste, [], 2);

    % Calcula porcentagem de acertos
    acertos = sum(classe_predita == classe_real);
 acuracia_atual = acertos / length(classe_real);
    historico_acuracias(i) = acuracia_atual;

    fprintf('Execucao %d: %.2f%%\n', i, acuracia_atual*100);
end

% Resultados Estatisticos
media_final = mean(historico_acuracias)*100;
desvio = std(historico_acuracias)*100;

fprintf('\nMedia Final: %.2f%%\n', media_final);
fprintf('Desvio Padrao: %.2f%%\n', desvio);


% Detalhes da ultima rodada (Visualizacao)
fprintf('\n--- Ultima Rodada (Amostra 15) ---\n');
nomes = {'Hernia', 'Spondilo', 'Normal'};

for k=1:15
    idx_real = classe_real(k);
    idx_ia = classe_predita(k);

  txt_real = nomes{idx_real};
    txt_ia = nomes{idx_ia};

    aviso = '';
    if idx_real ~= idx_ia
       aviso = ' <--- Erro';
    end
    fprintf('%02d | %s -> %s %s\n', k, txt_real, txt_ia, aviso);
end

% Matriz confusao (Mostra onde a rede errou mais)
fprintf('\nMatriz de Confusao:\n');
matriz = zeros(3,3);
for k=1:length(classe_real)
 r = classe_real(k);
    p = classe_predita(k);
    matriz(r,p) = matriz(r,p) + 1;
end
disp('    DH  SL  NO');
disp(matriz);
