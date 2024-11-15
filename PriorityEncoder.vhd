library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity Priority_Encoder is
  Port (
    D : in  STD_LOGIC_VECTOR(3 downto 0);
    Y : out STD_LOGIC_VECTOR(1 downto 0);
    V : out STD_LOGIC
  );
end Priority_Encoder;

architecture Behavioral of Priority_Encoder is
begin
  process (D)
  begin
    if D = "0000" then
      Y <= "00";
      V <= '0';  -- Nessun ingresso valido
    elsif D(3) = '1' then
      Y <= "11";
      V <= '1';
    elsif D(2) = '1' then
      Y <= "10";
      V <= '1';
    elsif D(1) = '1' then
      Y <= "01";
      V <= '1';
    elsif D(0) = '1' then
      Y <= "00";
      V <= '1';
    else
      Y <= "00"; -- Valore di default
      V <= '0';
    end if;
  end process;
end Behavioral;

