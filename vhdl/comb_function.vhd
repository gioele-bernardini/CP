entity comb_function is
port (
  -- a, b, c : in bit;
  a : in bit;
  b : in bit;
  c : in bit;

  z : out bit;
)
end entity comb_function;

architecture expression of comb_function is
  signal d : bit;
begin
  d <= a and b;
  z <= d or c;
end architecture expression;
  