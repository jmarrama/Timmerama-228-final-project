
oneshort = [];
twoshort = [];
i=1;
while i < length(fly.STIM)
    if fly.STIM(i) == 1
        if fly.STIM(i+4) ~= 1
            oneshort = [oneshort i];
        end
        i = i+5;
    end
    if fly.STIM(i)==2
        if fly.STIM(i+4) ~= 2
            twoshort=[twoshort i];
        end
        i=i+5;
    end
    i=i+1;
end

onegood = [];
twogood = [];
i=1;
while i < length(fly.STIM)
    if fly.STIM(i) == 1 && fly.STIM(i+6) == 1
        onegood = [onegood i];
        i = i+5;
    end
    if fly.STIM(i)==2 && fly.STIM(i+6) == 2
        twogood = [twogood i];
        i=i+5;
    end
    i=i+1;
end

lenone = zeros(10,1);
lentwo = zeros(10,1);
i = 1;
while i < length(fly.STIM)
    j=i+1;
    if fly.STIM(i) == 1
        while j < length(fly.STIM)
            if fly.STIM(j) ~= 1
                break;
            end
            j = j+1;
        end
        lenone(j-i) = lenone(j-i) + 1;
    end
    if fly.STIM(i) == 2
        while j < length(fly.STIM)
            if fly.STIM(j) ~= 2
                break;
            end
            j=j+1;
        end
        lentwo(j-i) = lentwo(j-i) + 1;
    end
    i=j;
end

for i=1:length(fly.indices)
   if any(fly.STIM(fly.indices{i})==1) && any(fly.STIM(fly.indices{i})==1)
       break;
   end
end