file = csvread('nuSVM_poly.csv');
nu = file(1:99, 1);
nu_rate_poly2 = file(1:99, 2);
nu_rate_poly3 = file(101:199, 2);
nu_rate_poly4 = file(201:299, 2);
file = csvread('nuSVM_rbf.csv');
nu_rate_rbf = file(1:99, 2);

file = csvread('nuSVM_gamma_poly.csv');
nu_rate_poly2g = file(1:99, 2);
nu_rate_poly3g = file(101:199, 2);
nu_rate_poly4g = file(201:299, 2);
file = csvread('nuSVM_gamma_rbf.csv');
nu_rate_rbfg = file(1:99, 2);

plot(nu, nu_rate_poly2, ...
     nu, nu_rate_poly2g, ...
     nu, nu_rate_poly3, ...
     nu, nu_rate_poly3g, ...
     nu, nu_rate_poly4, ...
     nu, nu_rate_poly4g, ...
    'LineWidth', 2)
legend('poly 2', 'poly 2 with gamma', ...
       'poly 3', 'poly 3 with gamma', ...
       'poly 4', 'poly 4 with gamma')
title('Overall Error Rate (\nu-SVM)')
xlabel('\nu')
ylabel('(%)')
ylim([1 100])

figure
plot(nu, nu_rate_rbf, ...
     nu, nu_rate_rbfg, ... 
     'LineWidth', 2)
legend('rbf', 'rbf with gamma')
title('Overall Error Rate (\nu-SVM)')
xlabel('\nu')
ylabel('(%)')
ylim([1 100])