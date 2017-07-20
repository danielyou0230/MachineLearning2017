%file = csvread('data_pca.csv');
file = csvread('lda.csv');
t = file(: , 3:3);
x = file(: , 1:1);
y = file(: , 2:2);

scatter(x, y, [], t, 'filled');
xlabel('x')
ylabel('y')
title('Normalised Test Data (LDA)')

file = csvread('norm.csv');
figure
t_g = file(: , 3:3);
x_g = file(: , 1:1);
y_g = file(: , 2:2);

scatter(x_g, y_g, [], t_g, 'filled');
xlabel('x')
ylabel('y')
title('Normalised Dataset (PCA)')