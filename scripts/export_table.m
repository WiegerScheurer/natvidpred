% export_table.m
load('/project/3018078.02/MEG_ingmar/ConditionTable.mat');
writetable(ConditionTable, '/project/3018078.02/natvidpred_workspace/ConditionTable.csv');