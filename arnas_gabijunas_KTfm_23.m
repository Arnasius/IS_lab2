clc;
clear;

% Įvestis ir norimas išvesties rezultatas
x = 0.1:1/22:1;
d = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2;

% Neuronų skaičius paslėptajame sluoksnyje
H = 6;

% Inicializuojame svorius ir poslinkius paslėptam sluoksniui
w_hidden = rand(1, H); % 1xH matrica svoriams nuo įvesties iki paslėptojo sluoksnio
b_hidden = rand(1, H); % 1xH matrica paslėptojo sluoksnio poslinkiams

% Inicializuojame svorius ir poslinkius išvesties sluoksniui
w_output = rand(H, 1); % Hx1 matrica svoriams nuo paslėptojo iki išvesties sluoksnio
b_output = rand(1);    % Poslinkis išvesties sluoksniui

eta = 0.01; % Mokymosi tempas

% Mokymas naudojant atgalinio sklidimo algoritmą
for epoch = 1:50000
    for i = 1:length(x)
        % Paslėptasis sluoksnis
        v_hidden = x(i) * w_hidden + b_hidden;
        y_hidden = tanh(v_hidden); % Hiperbolinė tangento aktyvavimo funkcija

        v_output = y_hidden * w_output + b_output;
        y_output = v_output; % Tiesinė aktyvavimo funkcija išvesties sluoksniui

        % Klaidos skaičiavimas
        e = d(i) - y_output;

        % Skaičiuojamas klaidos gradientas
        delta_output = e;
        delta_hidden = (1 - y_hidden.^2) .* (delta_output * w_output');

        % Atnaujiname svorius
        % Išėjimo sluoksnis
        w_output = w_output + eta * y_hidden' * delta_output;
        b_output = b_output + eta * delta_output;
        % Paslėptasis sluoksnis
        w_hidden = w_hidden + eta * x(i) * delta_hidden;
        b_hidden = b_hidden + eta * delta_hidden;
    end
end

% Testuojame MLP
X_test = 0.1:1/220:1;
Y_test = zeros(1, length(X_test));

for i = 1:length(X_test)
    v_hidden_test = X_test(i) * w_hidden + b_hidden;
    y_hidden_test = tanh(v_hidden_test);
    v_output_test = y_hidden_test * w_output + b_output;
    Y_test(i) = v_output_test;
end

% Braižome grafiką
plot(x, d, 'b', X_test, Y_test, 'g');
legend('Tikrasis', 'MLP Aproksimacija');
