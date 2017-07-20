file = csvread('distribution.csv');
x = file(:, 1);
y = file(:, 2);
c = file(:, 3);
b = file(:, 4);
sv = file(:, 5);

outlier_x = [];
outlier_y = [];
normal_x = [];
normal_y = [];
normal_c = [];
outlier_c = [];
sv_x = [];
sv_y = [];
sv_c = [];
dual_x = [];
dual_y = [];
dual_c = [];

for itr = 1:5000;
    if sv(itr) == 1
        sv_x = [sv_x x(itr)];
        sv_y = [sv_y y(itr)];
        sv_c = [sv_c c(itr)];
    end
    
    if b(itr) == 0
        normal_x = [normal_x x(itr)];
        normal_y = [normal_y y(itr)];
        normal_c = [normal_c c(itr)];
    end
    
    if b(itr) == 1
        outlier_x = [outlier_x x(itr)];
        outlier_y = [outlier_y y(itr)];
        outlier_c = [outlier_c c(itr)];
    end
    
    if b(itr) == 1 && sv(itr) == 1
        dual_x = [dual_x x(itr)];
        dual_y = [dual_y y(itr)];
        dual_c = [dual_c c(itr)];
    end
end

figure
scatter(normal_x, normal_y, [], normal_c, 'd', 'LineWidth', 1.5)
hold on
scatter(sv_x, sv_y, [], sv_c, 'o', 'LineWidth', 1.5)
hold on 

scatter(outlier_x, outlier_y, [], outlier_c, 'x', 'LineWidth', 2)
title('Training Data Distribution (normalised)')
xlim([0 1]);
ylim([0 1]);
xlabel('x')
ylabel('y')

figure
scatter(dual_x, dual_y, [], dual_c, 'o', 'LineWidth', 2)
hold on
scatter(dual_x, dual_y, [], dual_c, 'x', 'LineWidth', 2)
title('Training Data Distribution (normalised)')
legend('outliers', 'outliers who are also support vectors');
xlim([0 1]);
ylim([0 1]);
xlabel('x')
ylabel('y')