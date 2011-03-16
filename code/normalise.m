function [v s] = normalise(vec)

s = sum(vec);
v = vec ./ s;