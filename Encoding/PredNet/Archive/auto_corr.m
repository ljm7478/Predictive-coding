function [correlations, p_values] = auto_corr(y_true, y_pred)
    [n, m] = size(y_true);
    correlations = zeros(1, m);
    p_values = zeros(1, m);
    
    for i = 1:m
        [r, p] = corrcoef(y_true(:, i), y_pred(:, i));
        correlations(i) = r(1, 2);
        p_values(i) = p(1, 2);
    end
    
%     fprintf('Correlation and p-value for each column:\n');
%     for i = 1:m
%         fprintf('Column %d: correlation = %.2f, p-value = %.2f\n', i, correlations(i), p_values(i));
%     end
end