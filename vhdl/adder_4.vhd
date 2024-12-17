entity Adder4 is
port (
  A : in bit_vector(3 downto 0);
  B : in bit_vector(3 downto 0);
  cin : in bit;

  S : out bit_vector(3 downto 0);
  cout : out bit;
)

architecture structural of Adder4 is
  -- 1 elemento in meno che non servira' per FA3!
  signal C : bit_vector(3 downto 1);
begin
  FA0 : full_adder port map (A(0), B(0), cin, S(0), C(1));
  FA1 : full_adder port map (A(1), B(1), C(1), S(1), C(2));
  FA2 : full_adder port map (A(2), B(2), C(2), S(2), C(3));
  FA3 : full_adder port map (A(3), B(3), C(3), S(3), cout);
end architecture structural;