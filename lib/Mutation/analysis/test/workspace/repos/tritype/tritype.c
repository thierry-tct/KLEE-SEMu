
/*---------------------------------------------------------------------------*/
/*   
                  Program file name : tritype.c
                  Functions         : main
*/
/*---------------------------------------------------------------------------*/

/* include files */
#include <stdio.h>
#include <stdlib.h>

/*---------------------------------------------------------------------------*/
/*   
                  Function         : main
                  Parameters       : none
                  Input            : 3 integers : triangle sides
                  Output           : type of triangle 
                  Description      : MATCH IS OUTPUT FROM THE ROUTINE:
                                     TRIANG = 1 IF TRIANGLE IS SCALENE
                                     TRIANG = 2 IF TRIANGLE IS ISOSCELES
                                     TRIANG = 3 IF TRIANGLE IS EQUILATERAL
                                     TRIANG = 4 IF NOT A TRIANGLE 
*/
/*---------------------------------------------------------------------------*/

/* TCT */
int char2int(char * str)
{
  int x = 0;
  if (str[0] >= '0' && str[0] <= '9')
     return str[0] - '0';
  
  printf("char2int ERROR!");
  exit(1);
}

int str2int(char * str)
{
  int x = 0;
  if (str[0] >= '0' && str[0] <= '9')
  {  
      x = x*10 + str[0] - '0';
      if (str[1] >= '0' && str[1] <= '9')
      {
        x = x*10 + str[1] - '0';
        if (str[2] >= '0' && str[2] <= '9')
        {
          x = x*10 + str[1] - '0';
        }
      }
      return x;
  }
  else
  {
    printf("str2int ERROR!");
    exit(1);
  }
}


#define USE_ATOI 0
#define USE_STR2INT 1
#define USE_CHAR2INT 2

#define USE_INT_PARSER USE_ATOI

int main(int argc, char **argv)
{

        int i, j, k, triang;

        /*scanf("%d %d %d", &i, &j, &k);
        printf("33\n");*/
#if USE_INT_PARSER == USE_ATOI
        i = atoi(argv[1]);
        j = atoi(argv[2]);
        k = atoi(argv[3]);
#elif USE_INT_PARSER == USE_STR2INT
        i = str2int(argv[1]);
        j = str2int(argv[2]);
        k = str2int(argv[3]);
#elif USE_INT_PARSER == USE_CHAR2INT
        i = char2int(argv[1]);
        j = char2int(argv[2]);
        k = char2int(argv[3]);
#else
#error "Invalid USE_INT_PARSER"
#endif


        /*
       After a quick confirmation that it's a legal  
       triangle, detect any sides of equal length
*/

        if (( i <= 0 ) ||  (j <= 0)  ||  (k < 0))
        {
                triang = 4;
        }
        else
        {
                triang = 0;
                if (i == j) 
                        triang += 1;
                if (i == k) 
                        triang += 2;
                if (j == k) 
                        triang += 3;

                if ( triang == 0)
                {

                        /*
       Confirm it's a legal triangle before declaring
       it to be scalene
*/

                        if ( i+j <= k  || j+k <= i  || i+k < j) 
                                triang = 4;
                        else 
                                triang = 1;
                }
                else
                {
                        /*
       Confirm it's a legal triangle before declaring
       it to be isosceles or equilateral
*/
                        if (triang > 3) 
                                triang = 3;
                        else if (triang == 1 && i+j > k) 
                                 triang = 2;
                        else if (triang == 2 && i+k > j)
                                 triang = 2;
                        else if (triang == 3 && j+k > i)
                                 triang = 2;
                        else
                                 triang = 4;
                }
        } 
        printf(" triang = %d\n", triang);

}

