-- Porta AND a 2 ingressi
entity And2 is
  port (x, y : in bit; z : out bit);
end entity And2

-- ex1 (example1)
architecture ex1 of And2 is
begin
  z <= x and y;
end architecture ex1;

