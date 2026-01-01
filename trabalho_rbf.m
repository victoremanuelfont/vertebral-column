% --- Versaﬁo Limpa e Corrigida ---
clear; clc;

% 1. Carregar dados
% Verifica se o arquivo existe
if ~exist('column_3C.dat', 'file')
    error('Arquivo column_3C.dat nao encontrado.');
end

[c1, c2, c3, c4, c5, c6, texto] = textread('column_3C.dat', '%f %f %f %f %f %f %s');
X = [c1, c2, c3, c4, c5, c6];

% Limpar espacos no texto
texto = strtrim(texto);

% 2. Converter classes
y_temp = zeros(length(texto), 1);
y_temp(strcmp(texto, 'DH')) = 1;
y_temp(strcmp(texto, 'SL')) = 2;
y_temp(strcmp(texto, 'NO')) = 3;

% Verificar se tudo foi convertido
if any(y_temp == 0)
    error('Erro na conversao das classes. Verifique os nomes no arquivo.');
end

% One-Hot Encoding
num_classes = 3;
n_amostras = size(X, 1);
Y = zeros(n_amostras, num_classes);
for i = 1:n_amostras
    Y(i, y_temp(i)) = 1;
end

% 3. Configuracoes
num_execucoes = 10;
p_treino = 0.7;
num_neuronios = 15;
sigma = 40;
acuracias = zeros(num_execucoes, 1);

fprintf('Iniciando 10 execucoes...\n');

% 4. Loop Principal
for i = 1:num_execucoes

    % Embaralhar dados
    indices = randperm(n_amostras);
    X_emb = X(indices, :);
    Y_emb = Y(indices, :);

    % Dividir Treino e Teste
    qte_treino = round(n_amostras * p_treino);
    X_treino = X_emb(1:qte_treino, :);
    Y_treino = Y_emb(1:qte_treino, :);
    X_teste = X_emb(qte_treino+1:end, :);
    Y_teste = Y_emb(qte_treino+1:end, :);

    % --- Treinamento RBF ---
    % Escolher Centros
    idx_centros = randperm(size(X_treino, 1), num_neuronios);
    Centros = X_treino(idx_centros, :);

    % Matriz H (Treino)
    H = zeros(size(X_treino, 1), num_neuronios);
    for k = 1:num_neuronios
        dif = X_treino - Centros(k, :);
        dist_sq = sum(dif .^ 2, 2);
        H(:, k) = exp(-dist_sq / (2 * sigma^2));
    end
    H = [ones(size(X_treino, 1), 1), H]; % Bias

    % Calcular Pesos (W)
    W = pinv(H) * Y_treino;

    % --- Teste RBF ---
    % Matriz H (Teste)
    H_teste = zeros(size(X_teste, 1), num_neuronios);
    for k = 1:num_neuronios
        dif = X_teste - Centros(k, :);
        dist_sq = sum(dif .^ 2, 2);
        H_teste(:, k) = exp(-dist_sq / (2 * sigma^2));
    end
    H_teste = [ones(size(X_teste, 1), 1), H_teste]; % Bias

    % Previsao
    Y_est = H_teste * W;

    % Avaliacao
    [~, c_pred] = max(Y_est, [], 2);
    [~, c_real] = max(Y_teste, [], 2);

    acc = sum(c_pred == c_real) / length(c_real);
    acuracias(i) = acc;

    fprintf('Execucao %d: Acuracia = %.2f%%\n', i, acc*100);
end

% Resultado Final
fprintf('---------------------\n');
fprintf('Media Final: %.2f%%\n', mean(acuracias)*100);
fprintf('Desvio Padrao: %.2f%%\n', std(acuracias)*100);
