-- Esempio di architettura strutturale

entity Adder4 is
  port (
    A, B : in bit_vector(3 downto 0);
    cin : in bit;
  
    -- Uscite: un vettore a 4 bit, un bit singolo (il carry!)
    cout : out bit;
    S : out bit_vector(3 downto 0)
  );
end Adder4;

architecture structural of Adder4 is
  signal C : bit_vector(3 downto 1);
begin
  -- Associazione implicita (ordine posizionale come in C)
  FA0: FullAdder port map (A(0), B(0), cin, C(1), S(0));
  FA1: FullAdder port map (A(1), B(1), C(1), C(2), S(1));
  FA2: FullAdder port map (A(2), B(2), C(2), C(3), S(2));
  FA3: FullAdder port map (A(3), B(3), C(3), cout, S(3));
end architecture structural;


-- Test
entity TestAdder4 is
end entity;

architecture equations of TestAdder4 is
  signal A, B, S : bit_vector (3 downto 0);
  signal cin, cout : bit;
begin
  -- Device Under Test (dut)
  dut: entity work.Adder4(esplicita) port map(
    A => A, B => B,
    S => S,
    cin => cin,
    cout => cout
  );

  -- Definizione degli stimoli
  A <= "0000", "0010" after 100 ns, "1110" after 200 ns, "0110" after 300 ns;
  B <= "0000", "1010" after 50 ns, "1110" after 200 ns, "0110" after 300 ns;
  S <= "0000", "1010" after 50 ns, "0011" after 150 ns, "1100" after 250 ns;
end architecture;

