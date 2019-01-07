#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char **argv)
{
  int a,b;
  int x;
 
/*  printf("Enter two numbers\n");
  scanf("%d %d", &a, &b);*/
  a = str2int(argv[1]);
  b = str2int(argv[2]);
 
  if (a > b)
    x = 1;
  else
    x = 2;

  return x;
}
