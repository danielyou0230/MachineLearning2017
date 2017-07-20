file = csvread('nuSVM_linear.csv');
nu = file(:, 1);
nu_rate_linear = file(1:99, 2);
file = csvread('nuSVM_poly.csv');
nu_rate_poly2 = file(1:99, 2);
nu_rate_poly3 = file(101:199, 2);
nu_rate_poly4 = file(201:299, 2);
file = csvread('nuSVM_rbf.csv');
nu_rate_rbf = file(1:99, 2);
%total = file(:, 3);
%C1 = file(:, 4) / 500 * 100;
%C2 = file(:, 5) / 500 * 100;
%C3 = file(:, 6) / 500 * 100;
%C4 = file(:, 7) / 500 * 100;
%C5 = file(:, 8) / 500 * 100;

plot(nu, nu_rate_linear, ...
     nu, nu_rate_poly2, ...
     nu, nu_rate_poly3, ...
     nu, nu_rate_poly4, ...
     nu, nu_rate_rbf, ...
    'LineWidth', 2)

legend('linear', ...
       'poly 2', 'poly 3', 'poly4', ...
       'rbf')
   
title('Overall Error Rate (\nu-SVM)')
xlabel('\nu')
ylabel('(%)')
ylim([1 100])

%figure
%plot(nu, C1, nu, C2, nu, C3, ...
%    nu, C4, nu, C5, ...
%    'LineWidth', 2)
%legend('Class 1', 'Class 2', 'Class 3', ...
%    'Class 4', 'Class 5')
%title('In-Class Error Rate')
%xlabel('\nu')
%ylabel('(%)')
%ylim([-1 100])

file = csvread('cSVM_linear.csv');
c = file(:, 1);
c_rate_linear = file(1:20, 2);
file = csvread('cSVM_poly.csv');
c_rate_poly2 = file(1:20, 2);
c_rate_poly3 = file(21:40, 2);
c_rate_poly4 = file(41:60, 2);
file = csvread('cSVM_rbf.csv');
c_rate_rbf = file(1:20, 2);

figure
plot(c, c_rate_linear, ...
     c, c_rate_poly2, ...
     c, c_rate_poly3, ...
     c, c_rate_poly4, ...
     c, c_rate_rbf, ...
    'LineWidth', 2)

legend('linear', ...
       'poly 2', 'poly 3', 'poly4', ...
       'rbf')
   
title('Overall Error Rate (C-SVM)')
xlabel('exponent of C (base = 2)')
ylabel('(%)')