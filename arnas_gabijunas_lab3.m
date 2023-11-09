clc;
clear;

% Įvestis ir norimas išvesties rezultatas
x = 0.1:1/22:1;
d = (1 + 0.6 * sin(2 * pi * x / 0.7) + 0.3 * sin(2 * pi * x)) / 2;

% RBF centrų ir spindulių pasirinkimas
c1 = 0.65; % Centro c1 reikšmė
r1 = 0.1; % Spindulio r1 reikšmė
c2 = 0.5; % Centro c2 reikšmė
r2 = 0.1; % Spindulio r2 reikšmė

% Inicializuojame svorius ir poslinkį išvesties sluoksniui
w1 = rand(1); % Svoris nuo pirmo RBF iki išvesties
w2 = rand(1); % Svoris nuo antro RBF iki išvesties
w0 = rand(1); % Poslinkis išvesties sluoksniui

eta = 0.01; % Mokymosi tempas

% Mokymas naudojant perceptrono mokymo algoritmą
for epoch = 1:500000
    for i = 1:length(x)
        % Skaičiuojame RBF tinklo išėjimus
        f1 = exp(-(x(i)-c1)^2/(2*r1^2));
        f2 = exp(-(x(i)-c2)^2/(2*r2^2));

        % Išvesties sluoksnis
        y_output = w1 * f1 + w2 * f2 + w0;

        % Klaidos skaičiavimas
        e = d(i) - y_output;

        % Atnaujiname svorius
        w1 = w1 + eta * e * f1;
        w2 = w2 + eta * e * f2;
        w0 = w0 + eta * e;
    end
end

% Testuojame RBF tinklą
X_test = 0.1:1/220:1;
Y_test = zeros(1, length(X_test));

for i = 1:length(X_test)
    f1_test = exp(-(X_test(i)-c1)^2/(2*r1^2));
    f2_test = exp(-(X_test(i)-c2)^2/(2*r2^2));
    Y_test(i) = w1 * f1_test + w2 * f2_test + w0;
end

% Braižome grafiką
plot(x, d, 'b', X_test, Y_test, 'g');
legend('Tikrasis', 'RBFN Aproksimacija');
