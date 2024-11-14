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

