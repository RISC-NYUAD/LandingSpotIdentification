#ifndef NETWORKCLIENT_H
#define NETWORKCLIENT_H


#include <cstdint>
#include <string>
#include <memory>
#include <chrono>
#include <boost/asio.hpp>

//https://www.programmersought.com/article/58255953409/
/*# First delete the local original boost library
rm -f /usr/lib/libboost*
rm -fr 'find / -name libboost*'
# Remove boost header file
mv /usr/include/boost /usr/include/boost-bak
# Download wget
apt-get install wget
# Download Boost library
wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
tar -zxvf boost_1_66_0.tar.gz
cd boost_1_66_0
# Build scripts that meet the environment
./bootstrap.sh
# Compile you know
./b2 install
*/

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}




class NetworkClient
{
public:
    NetworkClient(const std::string &host, const std::string &port);
    NetworkClient(const NetworkClient&) = delete;
    NetworkClient& operator=(const NetworkClient&) = delete;
    ~NetworkClient();

    void connect();
    void disconnect();

    int write(const uint8_t *buffer, std::size_t size);
    int write(const std::string &buffer);

    int wait(std::chrono::steady_clock::duration timeout);
    int available();
    int read_some(uint8_t *buffer, std::size_t size);
    int read_some_wait(uint8_t *buffer, std::size_t size);
    int read_exactly(uint8_t *buffer, std::size_t size, std::chrono::steady_clock::duration timeout);
    int read_until(std::string &buffer, char delim, std::chrono::steady_clock::duration timeout);

    operator bool() const { return connected; }
    bool is_connected() const { return connected; }

    std::string get_host() const { return host; }
    std::string get_port() const { return port; }

    void set_debug(bool debug) { this->debug = debug;  }
    bool get_debug() const { return debug;  }

protected:
    bool connected = false;
    std::string host;
    std::string port;
    bool debug = false;

    std::shared_ptr<boost::asio::io_context> io_context;
    std::unique_ptr<boost::asio::ip::tcp::resolver> resolver;
    std::unique_ptr<boost::asio::ip::tcp::socket> socket;

    bool run_for(std::chrono::steady_clock::duration timeout); /// Run operation with timeout
    bool run_until(const std::chrono::steady_clock::time_point &timepoint); /// Run operation until timepoint
};

#endif // NETWORKCLIENT_H
