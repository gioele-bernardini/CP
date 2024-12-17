library IEEE;
use IEEE.std_logic_1164.all, IEEE.numeric_std.all;

entity Adder is
  generic (
    n : natural := 4;
  )

  port (
    a, b : in std_logic_vector(n-1 downto 0);
    cin : in std_logic;

    s : out std_logic_vector(n-1 downto 0);
    cout : out std_logic;
  )
end entity Adder;

architecture Unsigned of Adder is
  signal result, carry : unsigned(n downto 0);
  constant zeros : unsigned(n-1 downto 0) := (others => '0');
begin
  carry <= (zeros & cin);
  result <= ('0' & unsigned(a)) + ('0' & unsigned(b)) + carry;

  sum <= std_logic_vector(result(n-1 downto 0));
  cout <= result(n);
end architecture Unsigned;
