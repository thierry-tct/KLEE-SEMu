
/* include files */
#include <stdio.h>
#include <stdlib.h>

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

        int i,j;
        scanf("%d %d", &i, &j);
/*
#if USE_INT_PARSER == USE_ATOI
        i = atoi(argv[1]);
        j = atoi(argv[2]);
#elif USE_INT_PARSER == USE_STR2INT
        i = str2int(argv[1]);
        j = str2int(argv[2]);
#elif USE_INT_PARSER == USE_CHAR2INT
        i = char2int(argv[1]);
        j = char2int(argv[2]);
#else
#error "Invalid USE_INT_PARSER"
#endif
*/
        printf("Hello World! i=%d, j=%d\n", i,j);
}

