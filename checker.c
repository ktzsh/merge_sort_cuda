#include <stdio.h>
#include <time.h>

int main(int argc, char const *argv[]) {

    struct timespec start, stop;
    int n_list[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    int i, j;
    for(j=0; j<8; j++){
        printf("############ LENGTH OF LIST: %d ############\n", n_list[j]);
        int *sorted = (int *) malloc(n_list[j]*sizeof(int));
        int *list = (int *) malloc(n_list[j]*sizeof(int));
        int *sorted_s = (int *) malloc(n_list[j]*sizeof(int));
        int *list_s = (int *) malloc(n_list[j]*sizeof(int));
        for(i=0; i<n_list[j]; i++){
            list[i] = rand()%10000;
            list_s[i] = list[i];
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        mergesort(list, sorted, n_list[j]);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double result = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6;
        printf("TIME TAKEN(Parallel GPU): %fms\n", result);


        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        mergesort_cpu(list_s, sorted_s, n_list[j]);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        result = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6;
        printf("TIME TAKEN(Sequential CPU): %fms\n", result);

        for(i=1; i<n_list[j]; i++){
            if(sorted[i-1]>sorted[i]){
                printf("WRONG ANSWER _1\n");
                return -1;
            }
        }
        for(i=0; i<n_list[j]; i++){
            if(sorted_s[i]!=sorted[i]){
                printf("WRONG ANSWER _2\n");
                printf("P:%d, S:%d, Index:%d\n", sorted[i], sorted_s[i], i);
                return -1;
            }
        }
        printf("CORRECT ANSWER\n");

        free(sorted);
        free(list);
        free(sorted_s);
        free(list_s);
        printf("##################################################\n");
    }
    return 0;
}
