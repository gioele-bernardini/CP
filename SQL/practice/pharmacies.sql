create table az_pharm {
  name varchar(20),
  phone varchar(20),
  primary key (name),
};

create table pharm {
  name varchar(20),
  phone varchar(20),
  address varchar(20),
  primary key (name),
};

create table doctors {
  ssn: char(11),
  name: varchar(20),
  special varchar(20),
  dateb date -- datetime
}

