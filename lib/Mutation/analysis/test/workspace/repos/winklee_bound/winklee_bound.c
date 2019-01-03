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
 
  printf("Enter two numbers\n");
//  scanf("%d %d", &a, &b);
  char str[4];
  scanf ("%s", str);
  a = str2int(str);
  scanf ("%s", str);
  b = str2int(str);

  x = 0;
  if (a > 5)
    x += a;
  else
    x--;

  if (b > 5)
    x--;
  else
    x += b;

  return (int) (x > 5);
}
