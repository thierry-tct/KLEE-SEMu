#include <stdio.h>
 
int main(int argc, char **argv)
{
  int a,b;
  int x;
 
  printf("Enter two numbers\n");
  scanf("%d %d", &a, &b);
 
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
