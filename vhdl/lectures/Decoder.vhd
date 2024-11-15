library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity decoder_2_to_4 is
  Port (
    A : in STD_LOGIC_VECTOR(1 downto 0); -- Ingresso a 2 bit
    Y : out STD_LOGIC_VECTOR(3 downto 0) -- Uscita a 4 bit
  );
end decoder_2_to_4;

architecture Behavioral of decoder_2_to_4 is
begin
  with A select
    Y <= "0001" when "00",
         "0010" when "01",
         "0100" when "10",
         "1000" when "11",
         "0000" when others; -- Caso di Default
end Behavioral;

