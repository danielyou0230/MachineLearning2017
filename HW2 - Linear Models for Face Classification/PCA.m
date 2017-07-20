%file = csvread('data_pca.csv');
file = csvread('PCA_Dis.csv');
t = file(: , 3:3);
x = file(: , 1:1);
y = file(: , 2:2);

scatter(x, y, [], t, 'filled');
xlabel('x')
ylabel('y')
title('Transformed Data (Dis)')

%file = csvread('data_Gau.csv');
file = csvread('PCA_Phi.csv');
figure
%t_g = file(: , 3:3);
z_g = file(: , 3:3);
x_g = file(: , 1:1);
y_g = file(: , 2:2);

scatter3(x_g, y_g, z_g, [], t, 'filled');
xlabel('x')
ylabel('y')
title('Transformed Data (Phi)')
