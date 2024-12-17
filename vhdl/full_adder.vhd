entity full_adder is
port (
  a, b : in bit;
  cin : in bit;

  s : out bit;
  cout : out bit;
)

architecture equations of full_adder is
begin
  s <= a xor b xor cin;
  cout <= (a and b) or (a and cin) or (b and cin);
end architecture equations;