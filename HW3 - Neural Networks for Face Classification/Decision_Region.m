file = csvread('gen_1H.csv');
t_gen1h = file(: , 3:3);
x_gen1h = file(: , 1:1);
y_gen1h = file(: , 2:2);

scatter(x_gen1h, y_gen1h, [], t_gen1h, 'filled');
xlabel('x')
ylabel('y')
title('Decision Region (PCA normalised) 1 Hidden Layer')

file = csvread('gen_2H.csv');
figure
t_gen2h = file(: , 3:3);
x_gen2h = file(: , 1:1);
y_gen2h = file(: , 2:2);

scatter(x_gen2h, y_gen2h, [], t_gen2h, 'filled');
xlabel('x')
ylabel('y')
title('Decision Region (PCA normalised) 2 Hidden Layers')