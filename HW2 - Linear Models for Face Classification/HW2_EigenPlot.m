file = csvread('eigenvalues_Gen.csv');
%file = csvread('eigenvalues_Gen.csv');

eigval = file(: ,:);
eigval = reshape(eigval,1, 900);
val_plt = cumsum(eigval);
plot(val_plt, 'LineWidth',5)
ylim([0 100])
xlabel('Dimension after PCA')
ylabel('Information (%)')
title('Cumulative information (w.r.t dimension) acquired from PCA')