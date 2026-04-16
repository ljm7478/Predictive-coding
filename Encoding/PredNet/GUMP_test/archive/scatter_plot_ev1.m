base_dir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Wmapping/6layers_32channel/pca_result';

subjects = {'sub-01',  'sub-03',  'sub-05',  'sub-09',  'sub-14',  'sub-16',  'sub-18',  'sub-20', 'sub-02',  'sub-04',  'sub-06',  'sub-10',  'sub-15',  'sub-17',  'sub-19'};
features = {'Ahat', 'E'};

total_ev = zeros(15,2,5,8,30);

x = linspace(1,30,30);

for seg = 1:8
    for j = 1:2
        
        feat = features{j};
        
        for k = 1:5
            
            for i=1:15
                
                sub = subjects{i};
                ev = load(sprintf([base_dir, '/%s/%s_%s%s_seg-%s_explained.mat'], sub, sub, feat, num2str(k), num2str(seg))).explained;

                total_ev(i,j,k,seg,1:30) = ev(1:30);
            
            end
        end
    end
end

avg_ev = squeeze(mean(total_ev,2));
avg_ev = squeeze(mean(avg_ev,2));

for i=1:8
    
    figure;
    plot(x, squeeze(avg_ev(1,seg,:)),'-o');
    title(sprintf(['All subject - avg features - seg%s - 30 PCs'], num2str(seg)));  
    hold on;
    plot(x, squeeze(avg_ev(2,seg,:)),'-o');
    plot(x, squeeze(avg_ev(3,seg,:)),'-o');
    plot(x, squeeze(avg_ev(4,seg,:)),'-o');
    plot(x, squeeze(avg_ev(5,seg,:)),'-o');
    plot(x, squeeze(avg_ev(6,seg,:)),'-o');
    plot(x, squeeze(avg_ev(7,seg,:)),'-o');
    plot(x, squeeze(avg_ev(8,seg,:)),'-o');
    plot(x, squeeze(avg_ev(9,seg,:)),'-o');
    plot(x, squeeze(avg_ev(10,seg,:)),'-o');
    plot(x, squeeze(avg_ev(11,seg,:)),'-o');
    plot(x, squeeze(avg_ev(12,seg,:)),'-o');
    plot(x, squeeze(avg_ev(13,seg,:)),'-o');
    plot(x, squeeze(avg_ev(14,seg,:)),'-o');
    plot(x, squeeze(avg_ev(15,seg,:)),'-o');
                    
    legend('Sub 1','Sub 2','Sub 3','Sub 4','Sub 5','Sub 6','Sub 7','Sub 8','Sub 9','Sub 10','Sub 11','Sub 12','Sub 13','Sub 14','Sub 15');

end

figure;
scatter(x, total_ev(:,1,1,1));
title('All subject - Ahat1 - seg1');  

figure;
scatter(x, total_ev(:,1,2,1));
title('All subject - Ahat2 - seg1');  

figure;
scatter(x, total_ev(:,1,3,1));
title('All subject - Ahat3 - seg1');  

figure;
scatter(x, total_ev(:,1,4,1));
title('All subject - Ahat4 - seg1');  

figure;
scatter(x, total_ev(:,1,5,1));
title('All subject - Ahat5 - seg1');  

figure;
scatter(x, total_ev(:,2,1,1));
title('All subject - E1 - seg1');  

figure;
scatter(x, total_ev(:,2,2,1));
title('All subject - E2 - seg1');  

figure;
scatter(x, total_ev(:,2,3,1));
title('All subject - E3 - seg1');  

figure;
scatter(x, total_ev(:,2,4,1));
title('All subject - E4 - seg1');  

figure;
scatter(x, total_ev(:,2,5,1));
title('All subject - E5 - seg1');  

figure;
scatter(x, total_ev(:,1,1,1));
title('All subject - All features - seg1');  
hold on;
scatter(x, total_ev(:,1,2,1));
scatter(x, total_ev(:,1,3,1));
scatter(x, total_ev(:,1,4,1));
scatter(x, total_ev(:,1,5,1));
scatter(x, total_ev(:,2,1,1));
scatter(x, total_ev(:,2,2,1));
scatter(x, total_ev(:,2,3,1));
scatter(x, total_ev(:,2,4,1));
scatter(x, total_ev(:,2,5,1));
legend('Ahat1','Ahat2','Ahat3','Ahat4','Ahat5','E1','E2','E3','E4','E5');

