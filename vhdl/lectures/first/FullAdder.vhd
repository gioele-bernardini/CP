-- Generico full-adder

entity FullAdder is
  port (
    A, B, Cin : in bit;
    Cout, Sum : out bit
  );
end FullAdder;

architecture Behavioral of FullAdder is
begin
  Sum <= A xor B xor Cin;
  Cout <= (A and B) or (A and Cin) or (B and Cin);
end Behavioral;

