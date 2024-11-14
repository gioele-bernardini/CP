-- Porta AND a 2 ingressi
entity And2 is
  port (x, y : in bit; z : out bit);
end entity And2;

-- ex1 (example1)
architecture ex1 of And2 is
begin
  z <= x and y;
end architecture ex1;


-- Testbench
entity TestAnd2 is
end entity TestAnd2;

architecture simple of TestAnd2 is
  -- Segnali interni di interconnessione
  -- Intuitivamente, i *fili* a cui collegare il blocco di calcolo (la nostra AND in questo caso!)
  signal a, b, c : bit;
begin
  -- Istanza del modulo da testare
  g1: And2 port map (x => a, y => b, z => c);
  -- Definizione degli stimoli
  a <= '0', '1' after 100 ns, '0' after 200 ns;
  b <= '0', '1' after 150 ns, '0' after 200 ns;
end architecture simple;

